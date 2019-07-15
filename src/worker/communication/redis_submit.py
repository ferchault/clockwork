
def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--workpackages', type=str, help='', metavar='file')

    parser.add_argument('--redis-connect', help="connection to redis server")
    parser.add_argument('--redis-task', help="name of queue")

    args = parser.parse_args()
    # Get some conformations
    if args.redis_connect is not None:

        import os
        import rediscomm as rediswrap

        print("submitting jobs")
        # make queue
        tasks = rediswrap.Taskqueue(args.redis_connect, args.redis_task)

        # Submit job list

        f = open(args.workpackages)

        for i, line in enumerate(f):
            line = line.strip()
            tasks.insert(line)

        print("submitted")

        f.close()

    return

if __name__ == '__main__':
    main()
