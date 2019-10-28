import socket

def check(host, port, logger):
    with socket.socket() as sock:
        print("[check] connecting to %s port %s" % (host,port))
        try:
            sock.connect((host, port))
            logger.info(" # AliveCheck({}, {}) : connection SUCCESS.\n".format(host, port))
            data = 'Check'
            res = sock.sendall(data.encode('utf-8'))
            sock.close()
            if len(res) <= 0:
                logger.error(" # AliveCheck({}, {}) : failed to receive response".format(host, port))
                return False

        except socket.error as e:
            logger.error(" # AliveCheck({}, {}) : Exception {} : {}".format(host, port, e.args[0], e.args[1]))
            return False

        return True

if __name__ == '__main__':
    # ip_port = {'localhost': [50016, 50080]}
    port = [5004, 5005]
    for port in port:
        run_client('localhost', port)

