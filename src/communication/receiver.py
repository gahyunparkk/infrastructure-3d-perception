"""Run without CARLA for a portable smoke test, or with CARLA to update actors."""
import argparse
import socket
from .protocol import parse_states

class CarlaSink:
    def __init__(self, host, port, blueprint):
        import carla
        self.carla = carla
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        self.world = client.get_world()
        self.bp = self.world.get_blueprint_library().find(blueprint)
        self.actors = {}
    def update(self, states):
        for vid in set(self.actors)-set(states):
            self.actors.pop(vid).destroy()
        for vid,(x,y,yaw) in states.items():
            tf=self.carla.Transform(self.carla.Location(x=x,y=y,z=0.2),self.carla.Rotation(yaw=yaw))
            if vid in self.actors:
                self.actors[vid].set_transform(tf)
            else:
                actor=self.world.try_spawn_actor(self.bp,tf)
                if actor is not None:self.actors[vid]=actor
    def close(self):
        for actor in self.actors.values():actor.destroy()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bind',default='127.0.0.1')
    p.add_argument('--port',type=int,default=5001)
    p.add_argument('--carla',action='store_true',help='Requires a running CARLA server and its Python API')
    p.add_argument('--carla-host',default='127.0.0.1')
    p.add_argument('--carla-port',type=int,default=2000)
    p.add_argument('--blueprint',default='vehicle.tesla.model3')
    args=p.parse_args()
    sink=CarlaSink(args.carla_host,args.carla_port,args.blueprint) if args.carla else None
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        sock.bind((args.bind,args.port))
        print(f'Listening on UDP {args.bind}:{args.port} ({"CARLA" if sink else "dry-run"})',flush=True)
        try:
            while True:
                packet,addr=sock.recvfrom(65535)
                try:states=parse_states(packet)
                except (ValueError,UnicodeError) as e:
                    print(f'Ignored malformed packet from {addr}: {e}',flush=True);continue
                print(f'{addr}: {states}',flush=True)
                if sink:sink.update(states)
        except KeyboardInterrupt:pass
        finally:
            if sink:sink.close()
if __name__=='__main__':main()
