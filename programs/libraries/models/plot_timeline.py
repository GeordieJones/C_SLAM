import json
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Configuration - Use absolute pathing to be completely certain!
json_file_path = "nyu_performance_profile.json"
# Switch this to the absolute path of your compiled binary
binary_path = os.path.abspath("../train_distance")  

def resolve_hex_address(binary, address):
    if not os.path.exists(binary):
        print(f"[ERROR] Binary not found at: {binary}")
        return address
        
    try:
        # We pass '-p' for clean parsing, and use the binary itself
        result = subprocess.run(
            ["addr2line", "-f", "-C", "-e", binary, address],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split("\n")
        func_name = lines[0]

        # If addr2line can't read it, it returns '??'
        if func_name == "??" or not func_name:
            return address
        return func_name
    except Exception as e:
        print(f"[DEBUG EXECUTION ERROR]: {e}")
        return address

# 2. Load and parse the execution data
with open(json_file_path, "r") as f:
    data = json.load(f)

events = data[0]["execution_timeline"]
df = pd.DataFrame(events)

# 3. Resolve function names before aggregating
print(f"Targeting Binary at: {binary_path}")
print("Mapping hex addresses to debug symbols...")

unique_addresses = df["function_address"].unique()
address_map = {addr: resolve_hex_address(binary_path, addr) for addr in unique_addresses}

# Quick validation check in terminal
print("\n--- Map Sample Results ---")
for k, v in list(address_map.items())[:5]:
    print(f"Hex Pointer: {k} -> Resolved Symbol: {v}")
print("--------------------------\n")

df["function_name"] = df["function_address"].map(address_map)

# 4. Aggregate performance statistics using real names
stats = df.groupby("function_name").agg(
    call_count=("call_order", "count"),
    total_time_us=("duration_us", "sum"),
    avg_time_us=("duration_us", "mean")
).reset_index()

# Sort the data frames for clear presentation layouts
stats_by_calls = stats.sort_values(by="call_count", ascending=True)
stats_by_avg = stats.sort_values(by="avg_time_us", ascending=True)
stats_by_total = stats.sort_values(by="total_time_us", ascending=True)

# 5. Build the visual profile charts
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
fig, axes = plt.subplots(3, 1, figsize=(12, 18))
fig.suptitle("Neural Network Code Execution Profile", fontsize=16, fontweight='bold')

# Plot 1: Call Volumes
axes[0].barh(stats_by_calls["function_name"], stats_by_calls["call_count"], color="skyblue", edgecolor="black")
axes[0].set_title("Function Call Volume (Counts)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Total Call Invocations")

# Plot 2: Average Latencies (Added 'r' prefix to fix the string format syntax warnings)
axes[1].barh(stats_by_avg["function_name"], stats_by_avg["avg_time_us"], color="salmon", edgecolor="black")
axes[1].set_title("Average Call Latency (Microseconds)", fontsize=12, fontweight='bold')
axes[1].set_xlabel(r"Time ($\mu$s)")

# Plot 3: Resource Bottleneck Sinks
axes[2].barh(stats_by_total["function_name"], stats_by_total["total_time_us"], color="lightgreen", edgecolor="black")
axes[2].set_title("Total Cumulative Time Spent", fontsize=12, fontweight='bold')
axes[2].set_xlabel(r"Total Cumulative Budget Time ($\mu$s)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("named_network_profile.png", dpi=300)
plt.show()
