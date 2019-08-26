import os
import rediscomm
import gzip

def correct_userpath(filepath):
    return os.path.expanduser(filepath)

def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--redis-task', help="redis task name", default=None)
    parser.add_argument('--redis-connect', '--redis-connect-str', help="connection to str redis server", default=None)
    parser.add_argument('--redis-connect-file', help="connection to redis server", default="~/db/redis_connection")

    parser.add_argument('--jobfile', type=str, help='txt of jobs', metavar='file')

    args = parser.parse_args()

    #

    redis_task = args.redis_task

    if args.redis_connect is not None:
        redis_connection = args.redis_connection_str

    else:

        if "~" in args.redis_connect_file:
            args.redis_connect_file = correct_userpath(args.redis_connect_file)

        if not os.path.exists(args.redis_connect_file):
            print("error: redis connection not set and file does not exists")
            print("error: path", args.redis_connect_file)
            quit()

        with open(args.redis_connect_file, 'r') as f:
            redis_connection = f.read().strip()


    if redis_task is None:
        print("error: no task")
        quit()


    print("submitting jobs to", redis_task, "/", redis_connection)

    # make queue
    tasks = rediscomm.Taskqueue(redis_connection, redis_task)

    with open(args.jobfile) as f:
        lines = []
        for line in f:
            x = line.strip()
            x = gzip.compress(x.encode())
            lines.append(x)
        tasks.insert_batch(lines, iscompressed=True)

    print("done")

    return

if __name__ == '__main__':
    main()
