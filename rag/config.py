import ray

if __name__ == "__main__":
    print("----------------start config----------------")
    # get current gpu + cpu available in this Ray cluster
    ray.init()
    print(f"Number of GPUs: {ray.cluster_resources()['GPU']}")
    print(f"Number of CPUs: {ray.cluster_resources()['CPU']}")
    print("----------------end config----------------")