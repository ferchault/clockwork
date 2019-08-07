import os
import rediscomm

def correct_userpath(filepath):
    return os.path.expanduser(filepath)

def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--redis-task', help="redis task name", default=None)
    parser.add_argument('--redis-connect', '--redis-connect-str', help="connection to str redis server", default=None)
    parser.add_argument('--redis-connect-file', help="connection to redis server", default="~/db/redis_connection")

    parser.add_argument('--redis-requeue', help="requeue orhaned", action="store_true")

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


    tasks = rediscomm.Taskqueue(redis_connection, redis_task)

    if args.redis_requeue:
        tasks.requeue_orphaned()

    for project in sorted(tasks.discover_projects()):
        tasks.print_stats(project)

    return

if __name__ == '__main__':
    main()

