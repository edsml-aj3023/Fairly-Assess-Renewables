import pypsa
import numpy as np
import pandas as pd


def build_and_optimize_network(scenario_name, penalty_weights, subsampled_data, relevant_carriers,
                               base_marginal_costs, base_capital_costs,
                               operational_penalties, construction_penalties):
    """
    Builds, parameterizes, and optimizes a PyPSA power system network.

    This function constructs a multi-bus power network model based on specified
    scenario parameters. It calculates dynamic capital and marginal costs by
    applying penalty weights to base costs. The network is populated with
    buses, loads, generators (renewable and conventional), storage units, and
    transmission links. Finally, it runs a linear optimization to find the
    cost-minimal system configuration and dispatch.

    Inputs:
        scenario_name (str): The name of the scenario, used for logging and
            conditional logic (e.g., assertions).
        penalty_weights (dict): A dictionary mapping penalty types (e.g.,
            'reliability', 'environmental') to their corresponding weights.
        subsampled_data (pd.DataFrame): A DataFrame with time-series data for
            loads and renewable generator availability factors. The index must
            be a DatetimeIndex representing the snapshots.
        relevant_carriers (list): A list of strings for the energy carriers
            (e.g., 'AC', 'onwind') to be included in the network.
        base_marginal_costs (dict): A dictionary of baseline marginal costs
            (in €/MWh) for each technology carrier.
        base_capital_costs (dict): A dictionary of baseline capital costs
            (in €/MW) for each technology carrier.
        operational_penalties (dict): A nested dictionary defining additive
            marginal cost penalties for each technology and penalty type.
        construction_penalties (dict): A nested dictionary defining proportional
            capital cost penalty factors for each technology and penalty type.

    Returns:
        pypsa.Network: The optimized PyPSA network object containing results
            such as optimal capacities and dispatch.
    """
    print(f"\n--- Running Scenario: {scenario_name} ---")
    network = pypsa.Network()

    # Add all relevant energy carriers to the network
    for carrier_name in relevant_carriers:
        network.add("Carrier", carrier_name)

    # Configure network snapshots from the time-series data
    network.set_snapshots(subsampled_data.index)
    # Each snapshot represents an equal fraction of the year (8760 hours)
    weight_per_snapshot = 8760 / len(network.snapshots)
    network.snapshot_weightings = pd.DataFrame(
        {'generators': weight_per_snapshot,
         'loads': weight_per_snapshot,
         'stores': weight_per_snapshot,
         'objective': weight_per_snapshot},
        index=network.snapshots
    )
    
    # Add 9 AC buses representing different geographical zones
    for i in range(1, 10):
        network.add("Bus", f"z{i}", carrier="AC")

    # --- Calculate Dynamic Costs Based on Scenario Penalties ---
    mc_values, cap_cost_values = {}, {}
    all_costed_carriers = set(base_marginal_costs.keys()) | set(base_capital_costs.keys())

    for carrier in all_costed_carriers:
        # Marginal costs are increased additively by operational penalties
        base_mc = base_marginal_costs.get(carrier, 0)
        total_op_penalty = sum(
            operational_penalties.get(carrier, {}).get(p_type, 0) * penalty_weights.get(p_type, 0)
            for p_type in penalty_weights
        )
        mc_values[carrier] = base_mc + total_op_penalty

        # Capital costs are increased proportionally by construction penalties
        base_cc = base_capital_costs.get(carrier, 0)
        total_const_penalty_factor = sum(
            construction_penalties.get(carrier, {}).get(p_type, 0) * penalty_weights.get(p_type, 0)
            for p_type in penalty_weights
        )
        cap_cost_values[carrier] = base_cc * (1 + total_const_penalty_factor)

    # --- Sanity Check Block ---
    # For specific scenarios, this block verifies that the dynamic cost
    # calculations are producing the expected values.
    if scenario_name == 'Sweep_reliability_w100':
        
        # Verify Onshore Wind Capital Cost Calculation
        reliability_penalty_cc = construction_penalties.get('onwind', {}).get('reliability', 0)
        expected_cc = base_capital_costs['onwind'] * (1 + reliability_penalty_cc * 100)
        assert np.isclose(cap_cost_values['onwind'], expected_cc), \
            f"Onshore wind Capital Cost calculation is incorrect for {scenario_name}!"

        # Manually verify Onshore Wind Marginal Cost for this specific scenario.
        mc_penalty = operational_penalties.get('onwind', {}).get('reliability', 0)
        expected_mc = base_marginal_costs['onwind'] + (mc_penalty * 100)
        assert np.isclose(mc_values['onwind'], expected_mc), \
            "Onshore wind Marginal Cost is incorrect for the reliability scenario!"

    # --- Add All Network Components ---
    # Transmission Links (AC lines between zones)
    network.add("Link", "z1-z2", bus0="z1", bus1="z2", p_nom_extendable=True, carrier="AC", capital_cost=cap_cost_values["AC"])
    network.add("Link", "z2-z4", bus0="z2", bus1="z4", p_nom_extendable=True, carrier="AC", capital_cost=cap_cost_values["AC"])
    network.add("Link", "z3-z5", bus0="z3", bus1="z5", p_nom_extendable=True, carrier="AC", capital_cost=cap_cost_values["AC"])
    network.add("Link", "z4-z5", bus0="z4", bus1="z5", p_nom_extendable=True, carrier="AC", capital_cost=cap_cost_values["AC"])
    network.add("Link", "z5-z6", bus0="z5", bus1="z6", p_nom_extendable=True, carrier="AC", capital_cost=cap_cost_values["AC"])
    network.add("Link", "z6-z7", bus0="z6", bus1="z7", p_nom_extendable=True, carrier="AC", capital_cost=cap_cost_values["AC"])
    network.add("Link", "z7-z8", bus0="z7", bus1="z8", p_nom_extendable=True, carrier="AC", capital_cost=cap_cost_values["AC"])
    network.add("Link", "z8-z9", bus0="z8", bus1="z9", p_nom_extendable=True, carrier="AC", capital_cost=cap_cost_values["AC"])
    
    # Electrical Loads
    GB_load = subsampled_data['Demand']
    # These fractions distribute the total Great Britain load across the 9 zones
    load_fractions = [0.0053, 0.0112, 0.0023, 0.0082, 0.0479, 0.0701, 0.1544, 0.5898, 0.1109]
    for i, frac in enumerate(load_fractions, 1):
        network.add("Load", f"load_z{i}", bus=f"z{i}", p_set=GB_load * frac)

    # Renewable Generators (with extendable capacity)
    for i in range(1, 10):
        network.add("Generator", f"Onshore Wind z{i}", bus=f"z{i}", p_nom_extendable=True, capital_cost=cap_cost_values["onwind"], p_max_pu=subsampled_data[f'W_On_Z{i}'], carrier="onwind", marginal_cost=mc_values["onwind"])
        if f'W_Off_Z{i}' in subsampled_data:
             network.add("Generator", f"Offshore Wind z{i}", bus=f"z{i}", p_nom_extendable=True, capital_cost=cap_cost_values["offwind"], p_max_pu=subsampled_data[f'W_Off_Z{i}'], carrier="offwind", marginal_cost=mc_values["offwind"])
    for i in [1, 3, 5, 7, 9]:
        network.add("Generator", f"Tidal Stream z{i}", bus=f"z{i}", p_nom_extendable=True, capital_cost=cap_cost_values["tidal"], p_max_pu=subsampled_data[f'Z{i}_tidal'], carrier="tidal", marginal_cost=mc_values["tidal"])

    # Storage Units (with extendable capacity)
    for i in range(1, 10):
        network.add("StorageUnit", f"Battery z{i}", bus=f"z{i}", p_nom_extendable=True, max_hours=2, carrier="battery", capital_cost=cap_cost_values["battery"], marginal_cost=mc_values["battery"])
    
    network.add("StorageUnit", "Pumped hydro z2 Foyers", bus="z2", p_nom_extendable=True, max_hours=16, carrier="hydroelectric_ph", capital_cost=cap_cost_values["hydroelectric_ph"], marginal_cost=mc_values["hydroelectric_ph"])
    network.add("StorageUnit", "Pumped hydro z2 Coire Glas", bus="z2", p_nom_extendable=True, max_hours=17, carrier="hydroelectric_ph", capital_cost=cap_cost_values["hydroelectric_ph"], marginal_cost=mc_values["hydroelectric_ph"])
    network.add("StorageUnit", "Pumped hydro z4", bus="z4", p_nom_extendable=True, max_hours=13, carrier="hydroelectric_ph", capital_cost=cap_cost_values["hydroelectric_ph"], marginal_cost=mc_values["hydroelectric_ph"])
    network.add("StorageUnit", "Pumped hydro z7 Dinorwig", bus="z7", p_nom_extendable=True, max_hours=5, carrier="hydroelectric_ph", capital_cost=cap_cost_values["hydroelectric_ph"], marginal_cost=mc_values["hydroelectric_ph"])
    network.add("StorageUnit", "Pumped hydro z7 Ffestiniog", bus="z7", p_nom_extendable=True, max_hours=12, carrier="hydroelectric_ph", capital_cost=cap_cost_values["hydroelectric_ph"], marginal_cost=mc_values["hydroelectric_ph"])
    #network.add("StorageUnit", "DSR z8", bus="z8", p_nom_extendable=False, max_hours=2, carrier="DSR", marginal_cost=mc_values["DSR"], capital_cost=cap_cost_values["DSR"])
    network.add("StorageUnit", "DSR z8", bus="z8", p_nom=8930, max_hours=2, carrier="DSR",p_min_pu=subsampled_data['DSR_min'], p_max_pu=subsampled_data['DSR_max'], marginal_cost=mc_values["DSR"], standing_loss=0.1)
    # Load Shedding Generators (last resort to meet demand)
    for i in range(1, 10):
        network.add("Generator", f"Load shed z{i}", bus=f"z{i}", p_nom=5000,
                    p_nom_extendable=False, carrier="load_shed",
                    # Load shedding cost is a high, fixed value not affected by penalties.
                    marginal_cost=base_marginal_costs["load_shed"])

    # --- Run Optimization ---
    # Solve the linear problem to minimize total system cost
    network.optimize(solver_name="highs")
    print(f"Optimization status for {scenario_name}: {network.model.status}")
    
    return network


def extract_detailed_results(network, scenario_name, base_capital_costs, base_marginal_costs):
    """
    Extracts key results and calculates the true economic cost from an optimized PyPSA network.

    This function processes a solved PyPSA network object to derive meaningful outputs.
    Its primary role is to calculate the "true" economic system cost using the original,
    unpenalized base costs, separating it from the optimization objective function,
    which includes non-economic penalties. It also computes and records various
    Key Performance Indicators (KPIs) such as total generation, installed capacities by
    technology, generation mix percentages, and energy curtailment.

    Inputs:
        network (pypsa.Network): The optimized PyPSA network object containing solver results.
        scenario_name (str): A unique identifier for the optimization scenario being processed.
        base_capital_costs (dict): A dictionary of the original, unpenalized capital
            costs (€/MW) for each technology.
        base_marginal_costs (dict): A dictionary of the original, unpenalized marginal
            costs (€/MWh) for each technology.

    Returns:
        dict: A dictionary containing the scenario name and a comprehensive set of
              calculated results and KPIs. Returns a minimal dictionary with a 'Failed'
              status if the optimization did not solve successfully.
    """
    # Initialize a dictionary to hold all results for this scenario
    results_dict = {'Scenario': scenario_name}

    # If the network has no 'objective' attribute, the optimization likely failed.
    # Exit early to prevent errors.
    if not hasattr(network, 'objective'):
        print(f"Skipping results for failed scenario: {scenario_name}")
        results_dict['Economic_System_Cost_GBP'] = 'Failed'
        return results_dict

    print(f"\n--- Extracting Results for: {scenario_name} ---")

    # Helper function to convert a lump-sum capital cost into an equivalent annual cost
    def annualize_cost(lifetime_cost, lifetime=25, discount_rate=0.05):
        """Calculates the equivalent annual cost from a total investment cost."""
        if lifetime_cost == 0: return 0
        # Standard capital recovery factor formula
        return lifetime_cost * (discount_rate * (1 + discount_rate)**lifetime) / ((1 + discount_rate)**lifetime - 1)

    # Convert all base capital costs to their annualized equivalents for fair comparison
    base_capital_costs_annualized = {k: annualize_cost(v) for k, v in base_capital_costs.items()}
    
    # Get the snapshot weights, which represent the duration of each time step in hours
    gen_weights = network.snapshot_weightings.generators
    store_weights = network.snapshot_weightings.stores
    
    # Initialize the true economic cost, which will be calculated from base costs
    economic_cost = 0

    # --- 1. Calculate True Operational Costs (based on unpenalized marginal costs) ---
    # The 'load_shed' cost is reverted back to its original value from the evolve study
    true_base_marginal_costs = base_marginal_costs.copy()
    true_base_marginal_costs['load_shed'] = 3000
    
    # Calculate total energy dispatch (MWh) for each generator over the year
    gen_dispatch_mwh = network.generators_t.p.mul(gen_weights, axis=0)
    # Calculate total energy flow (MWh) through each storage unit (charge and discharge)
    storage_flow_mwh = network.storage_units_t.p.abs().mul(store_weights, axis=0)

    # Sum the dispatch and flow by technology carrier
    gen_dispatch_by_carrier = gen_dispatch_mwh.T.groupby(network.generators.carrier).sum().T.sum()
    storage_flow_by_carrier = storage_flow_mwh.T.groupby(network.storage_units.carrier).sum().T.sum()

    # Calculate the total operational cost by multiplying energy produced by base marginal cost
    for carrier, base_mc in true_base_marginal_costs.items():
        total_dispatch = gen_dispatch_by_carrier.get(carrier, 0) + storage_flow_by_carrier.get(carrier, 0)
        economic_cost += total_dispatch * base_mc

    # --- 2. Calculate True Capital Costs (Annualized, based on unpenalized capital costs) ---
    # Iterate through all component types that can have their capacity extended
    for component_list in [network.generators, network.storage_units, network.links]:
        extendable_assets = component_list[component_list.p_nom_extendable]
        # For each asset, add its annualized capital cost to the total economic cost
        for idx, asset in extendable_assets.iterrows():
            base_cc_annualized = base_capital_costs_annualized.get(asset.carrier, 0)
            economic_cost += asset.p_nom_opt * base_cc_annualized
    
    # Store the results
    results_dict['Optimization_Objective'] = network.objective
    results_dict['Economic_System_Cost_GBP'] = economic_cost

    # --- Energy Balance Sanity Check ---
    # Ensure the solver produced a physically valid result where energy supplied
    # nearly equals energy withdrawn.
    total_supply_mwh = network.statistics.supply().sum().sum()
    total_withdrawal_mwh = network.statistics.withdrawal().sum().sum()
    discrepancy = total_supply_mwh - total_withdrawal_mwh
    discrepancy_percent = 100 * abs(discrepancy) / total_supply_mwh if total_supply_mwh > 0 else 0
    
    assert discrepancy_percent < 0.1, (
            f"Energy balance discrepancy is high for scenario '{scenario_name}'. "
            f"Supply: {total_supply_mwh:,.2f} MWh, Withdrawal: {total_withdrawal_mwh:,.2f} MWh, "
            f"Discrepancy: {discrepancy_percent:.4f}%"
        )
        
    print(f"   Energy balance is within tolerance ({discrepancy_percent:.4f}%).")

    # --- Extract Other Key Performance Indicators (KPIs) ---

    # --- KPI: Installed Capacity (Optimized Assets) ---
    # Select only the assets that the optimizer was allowed to build (p_nom_extendable=True)
    extendable_gens = network.generators[network.generators.p_nom_extendable]
    extendable_stores = network.storage_units[network.storage_units.p_nom_extendable]
    extendable_links = network.links[network.links.p_nom_extendable]
    
    # Concatenate the optimized capacities (p_nom_opt) and carriers from these extendable assets
    p_nom_opts = pd.concat([
        extendable_gens.p_nom_opt,
        extendable_stores.p_nom_opt,
        extendable_links.p_nom_opt
    ])
    all_carriers = pd.concat([
        extendable_gens.carrier,
        extendable_stores.carrier,
        extendable_links.carrier
    ])
    
    # Group by carrier to find total installed capacity (MW) for each technology
    installed_cap = p_nom_opts.groupby(all_carriers).sum()
    for carrier, cap_value in installed_cap.items():
        if cap_value > 0.1: # Only record capacities that are non-trivial
            results_dict[f'Installed_{carrier}_MW'] = cap_value

    # --- KPI: Fixed Capacity (Non-Extendable Assets) ---
    # Select assets with pre-defined capacity (p_nom_extendable=False)
    fixed_gens = network.generators[~network.generators.p_nom_extendable]
    fixed_stores = network.storage_units[~network.storage_units.p_nom_extendable]
    
    # Use p_nom for fixed assets, not p_nom_opt
    fixed_p_noms = pd.concat([fixed_gens.p_nom, fixed_stores.p_nom])
    fixed_carriers = pd.concat([fixed_gens.carrier, fixed_stores.carrier])
    
    # Group by carrier to find total fixed capacity
    fixed_cap = fixed_p_noms.groupby(fixed_carriers).sum()
    for carrier, cap_value in fixed_cap.items():
        if cap_value > 0.1:
             # Use a different column name to distinguish fixed from optimized capacity
            results_dict[f'Fixed_Capacity_{carrier}_MW'] = cap_value

    # --- KPI: Total Generation & Generation Mix ---
    # Calculate total MWh generated by storage units (dispatch only, not charging)
    storage_generation_mwh = network.storage_units_t.p.clip(upper=0).abs().mul(store_weights, axis=0)
    # Sum all sources to get total generation to meet load
    total_sources_mwh = gen_dispatch_mwh.sum().sum() + storage_generation_mwh.sum().sum()
    results_dict['Total_Generation_MWh'] = total_sources_mwh

    # Calculate generation mix as a percentage of total generation
    storage_generation_by_carrier = storage_generation_mwh.T.groupby(network.storage_units.carrier).sum().T.sum()
    total_dispatch_by_carrier = gen_dispatch_by_carrier.add(storage_generation_by_carrier, fill_value=0)
    for carrier, gen_value in total_dispatch_by_carrier.items():
        results_dict[f'Gen_{carrier}_%'] = (gen_value / total_sources_mwh) * 100 if total_sources_mwh > 0 else 0

    # --- KPI: Curtailment ---
    # Calculate total and per-carrier curtailment using PyPSA's built-in statistics function
    curtailment_df = network.statistics.curtailment(aggregate_groups="sum")
    results_dict['Total_Curtailment_MWh'] = curtailment_df.sum()
    for carrier in network.carriers.index:
        curt_col_gen = f"Curtailment_('Generator', '{carrier}')_MWh"
        curt_col_store = f"Curtailment_('StorageUnit', '{carrier}')_MWh"
        results_dict[curt_col_gen] = curtailment_df.get(('Generator', carrier), 0)
        results_dict[curt_col_store] = curtailment_df.get(('StorageUnit', carrier), 0)
        
    return results_dict
