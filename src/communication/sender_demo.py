"""Tiny synthetic traffic-state sender, no ML model required."""
import argparse,socket,time,math
from .protocol import encode_states

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=5001)
    p.add_argument('--frames',type=int,default=20)
    p.add_argument('--interval',type=float,default=0.1)
    args=p.parse_args()
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        for i in range(args.frames):
            states={1:(1.0+i*0.1,3.0,0.0),2:(4.0,2.0+math.sin(i/5),20.0)}
            sock.sendto(encode_states(states),(args.host,args.port))
            time.sleep(args.interval)
    print(f'Sent {args.frames} synthetic frames')
if __name__=='__main__':main()
