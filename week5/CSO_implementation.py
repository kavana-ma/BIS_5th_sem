import numpy as np

# Number of tasks and VMs
num_tasks = 10
num_vms = 3

# Random execution times for each task (units arbitrary)
task_times = np.random.randint(1, 10, size=num_tasks)

# Fitness function: minimize makespan (max load on any VM)
def fitness(schedule):
    vm_loads = np.zeros(num_vms)
    for task, vm in enumerate(schedule):
        vm_loads[vm] += task_times[task]
    return max(vm_loads)

# Generate initial population (nests) of schedules
def init_nests(num_nests):
    return [np.random.randint(0, num_vms, num_tasks) for _ in range(num_nests)]

# Lévy flight-inspired perturbation: randomly reassign a few tasks
def levy_flight(schedule, alpha=1):
    new_schedule = schedule.copy()
    num_changes = np.random.randint(1, 5)  # Increase max changes
    for _ in range(num_changes):
        task_to_change = np.random.randint(0, num_tasks)
        new_vm = np.random.randint(0, num_vms)
        new_schedule[task_to_change] = new_vm
    return new_schedule

# Cuckoo Search Algorithm
def cuckoo_search(num_nests=15, max_iter=50, p=0.25):
    nests = init_nests(num_nests)
    fitnesses = np.array([fitness(n) for n in nests])
    
    best_idx = np.argmin(fitnesses)
    best_nest = nests[best_idx]
    best_fit = fitnesses[best_idx]

    for iteration in range(1, max_iter + 1):
        # Generate new solutions via Lévy flights
        for i in range(num_nests):
            new_nest = levy_flight(nests[i])
            new_fit = fitness(new_nest)
            if new_fit < fitnesses[i]:
                nests[i] = new_nest
                fitnesses[i] = new_fit
                if new_fit < best_fit:
                    best_nest = new_nest
                    best_fit = new_fit
        
        # Discovery step: randomly replace some nests
        for i in range(num_nests):
            if np.random.rand() < p:
                nests[i] = np.random.randint(0, num_vms, num_tasks)
                fitnesses[i] = fitness(nests[i])
                if fitnesses[i] < best_fit:
                    best_nest = nests[i]
                    best_fit = fitnesses[i]

        # Print best makespan at current iteration
        print(f"Iteration {iteration}: Best Makespan = {best_fit}")

    return best_nest, best_fit

# Run the CSA task scheduling
best_schedule, best_makespan = cuckoo_search(num_nests=30, max_iter=10, p=0.25)
print("Best Schedule (task->VM):", best_schedule)
print("Minimum Makespan:", best_makespan)
