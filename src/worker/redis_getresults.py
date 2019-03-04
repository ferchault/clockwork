
def main():

    import argparse
    import os
    import rediscomm as rediswrap

    parser = argparse.ArgumentParser()

    parser.add_argument('--connect-redis', help="connection to redis server")
    parser.add_argument('-f', '--filename', help="")

    args = parser.parse_args()

    tasks = rediswrap.Taskqueue(args.connect_redis, 'DEBUG')

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
