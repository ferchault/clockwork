
def main():

    import argparse
    import os
    import rediscomm as rediswrap

    parser = argparse.ArgumentParser()

    parser.add_argument('--redis-connect', help="connection to redis server")
    parser.add_argument('--redis-task', help="name of queue")
    parser.add_argument('-f', '--filename', help="")

    args = parser.parse_args()

    tasks = rediswrap.Taskqueue(args.redis_connect, args.redis_task)

    # get_results

    results = tasks.get_results()

    f = open(args.filename, 'w')
    for line in results:
        f.write(line)
        f.write("\n")

    f.close()


    return

if __name__ == '__main__':
    main()
