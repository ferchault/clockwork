
def main():

    import argparse
    import os
    import rediscomm as rediswrap

    parser = argparse.ArgumentParser()

    parser.add_argument('--redis-connect', help="connection to redis server")
    parser.add_argument('--redis-task', help="name of queue")
    args = parser.parse_args()

    tasks = rediswrap.Taskqueue(args.redis_connect, args.redis_task)

    for project in sorted(tasks.discover_projects()):
            tasks.print_stats(project)

    return

if __name__ == '__main__':
    main()

