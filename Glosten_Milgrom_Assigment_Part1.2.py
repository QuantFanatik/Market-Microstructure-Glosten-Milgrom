import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go


def run_simulation(pi, V_high=102, V_low=98, periods=100, initial_theta=0.5):
    # Create a DataFrame to store results
    results = pd.DataFrame(index=range(1, periods + 1),
                           columns=['Theta', 'mu_t', 'Sb', 'Sa', 'Bid',
                                    'Ask', 'Trader Type', 'Trade', 'Price',
                                    'Spread Size', 'Pricing Error'])
    theta = initial_theta

    for round in range(1, periods + 1):
        # Calculate expected value and bid/ask spreads
        mu_t = theta * V_high + (1 - theta) * V_low
        Sb = (pi * theta * (1 - theta) / (pi * theta + (1 - pi) / 2)) * (V_high - V_low)
        Sa = Sb  # Same calculation for both spreads in this case
        bid = mu_t - Sb
        ask = mu_t + Sa

        # Simulate trader arrival
        trader_type = 'I' if np.random.rand() < pi else 'U'

        # Determine trade direction based on trader type
        if trader_type == 'I':
            # Informed traders know the true value (V_low), so they sell if the price is above V_low, otherwise buy
            trade = -1 if bid > V_low else 1
        else:
            # Uninformed traders randomly choose buy or sell
            trade = 1 if np.random.rand() < 0.5 else -1

        # Determine trade price
        price = ask if trade == 1 else bid

        # Update the market maker's belief (theta)
        if trade == 1:
            theta = ((1 + pi) / 2 * theta) / (pi * theta + (1 - pi) / 2)
        else:
            theta = ((1 - pi) / 2 * theta) / (pi * (1 - theta) + (1 - pi) / 2)

        # Record the results
        spread_size = ask - bid
        pricing_error = abs(price - V_low)  # Adjusted to use V_low as the "true value"
        results.loc[round] = [theta, mu_t, Sb, Sa, bid, ask, trader_type, trade, price, spread_size, pricing_error]

    return results


# Parameters for the simulation
pi = 0.3  # Probability of encountering an informed trader
V_high = 102
V_low = 98
periods = 100  # Define periods globally

# Run the simulation
simulation_results = run_simulation(pi, V_high, V_low, periods)

# Save the results to CSV
csv_filename = 'glosten_milgrom_simulation_results.csv'
simulation_results.to_csv(csv_filename, index_label='Round')
print(f"Data saved to {csv_filename}")

# Plotting the results
plt.figure(figsize=(15, 5))

# Plot for Theta (Belief in High Value)
plt.subplot(1, 3, 1)
plt.plot(simulation_results.index, simulation_results['Theta'], color='blue')
plt.title('Evolution of Theta')
plt.xlabel('Round')
plt.ylabel('Theta')

# Plot for Spread Size
plt.subplot(1, 3, 2)
plt.plot(simulation_results.index, simulation_results['Spread Size'], color='green')
plt.title('Spread Size Over Time')
plt.xlabel('Round')
plt.ylabel('Spread Size')

# Plot for Pricing Error
plt.subplot(1, 3, 3)
plt.plot(simulation_results.index, simulation_results['Pricing Error'], color='red')
plt.title('Pricing Errors Over Time')
plt.xlabel('Round')
plt.ylabel('Pricing Error')

plt.tight_layout()
plt.show()

# Additional analysis with varying informed trader probabilities
informed_traders_range = np.linspace(0.1, 0.9, 10)
periods_range = list(range(1, periods + 1))

# Create matrices to store the results for 3D plotting
spread_matrix = np.zeros((len(informed_traders_range), periods))
pricing_error_matrix = np.zeros((len(informed_traders_range), periods))

# Loop over different proportions of informed traders
for idx, pi in enumerate(informed_traders_range):
    sim_results = run_simulation(pi, V_high, V_low, periods)
    spread_matrix[idx, :] = sim_results['Spread Size']
    pricing_error_matrix[idx, :] = sim_results['Pricing Error']

# Create 3D interactive plot for spread size
spread_fig = go.Figure(data=[go.Surface(
    z=spread_matrix,
    x=periods_range,
    y=informed_traders_range,
    colorscale='Viridis'
)])
spread_fig.update_layout(
    title='Spread Size over Time and Proportion of Informed Traders',
    scene=dict(
        xaxis_title='Periods',
        yaxis_title='Proportion of Informed Traders',
        zaxis_title='Spread Size'
    )
)
spread_fig.show()

# Create 3D interactive plot for pricing errors
pricing_error_fig = go.Figure(data=[go.Surface(
    z=pricing_error_matrix,
    x=periods_range,
    y=informed_traders_range,
    colorscale='Plasma'
)])
pricing_error_fig.update_layout(
    title='Pricing Error over Time and Proportion of Informed Traders',
    scene=dict(
        xaxis_title='Periods',
        yaxis_title='Proportion of Informed Traders',
        zaxis_title='Pricing Error'
    )
)
pricing_error_fig.show()

# Part 3: Simulate adverse selection with spread adjustment factors
informed_traders_range_adverse = np.linspace(0.1, 0.9, 10)
spread_adjustment_range = np.linspace(0.8, 1.2, 5).tolist()

# Create matrices for adverse selection results
spread_matrix_adverse = np.zeros((len(informed_traders_range_adverse), len(spread_adjustment_range)))
pricing_error_matrix_adverse = np.zeros((len(informed_traders_range_adverse), len(spread_adjustment_range)))

# Loop over different proportions of informed traders and spread adjustments
for idx, pi in enumerate(informed_traders_range_adverse):
    for jdx, spread_factor in enumerate(spread_adjustment_range):
        sim_results = run_simulation(pi, V_high, V_low, periods)
        adjusted_spread = sim_results['Spread Size'] * spread_factor
        adjusted_error = sim_results['Pricing Error']
        spread_matrix_adverse[idx, jdx] = adjusted_spread.mean()
        pricing_error_matrix_adverse[idx, jdx] = adjusted_error.mean()

# Create 3D interactive plot for spread size under adverse selection scenarios
spread_adverse_fig = go.Figure(data=[go.Surface(
    z=spread_matrix_adverse,
    x=spread_adjustment_range,
    y=informed_traders_range_adverse,
    colorscale='Viridis'
)])
spread_adverse_fig.update_layout(
    title='Adverse Selection: Spread Size under Modified Bid-Ask Spread',
    scene=dict(
        xaxis_title='Spread Adjustment Factor',
        yaxis_title='Proportion of Informed Traders',
        zaxis_title='Average Spread Size'
    )
)
spread_adverse_fig.show()

# Create 3D interactive plot for pricing errors under adverse selection scenarios
pricing_error_adverse_fig = go.Figure(data=[go.Surface(
    z=pricing_error_matrix_adverse,
    x=spread_adjustment_range,
    y=informed_traders_range_adverse,
    colorscale='Plasma'
)])
pricing_error_adverse_fig.update_layout(
    title='Adverse Selection: Pricing Error under Modified Bid-Ask Spread',
    scene=dict(
        xaxis_title='Spread Adjustment Factor',
        yaxis_title='Proportion of Informed Traders',
        zaxis_title='Average Pricing Error'
    )
)
pricing_error_adverse_fig.show()
