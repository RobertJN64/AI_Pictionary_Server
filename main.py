import multiprocessing
import server
import qd_ai as ai
#import ai



if __name__ == '__main__':
    mm = multiprocessing.Manager()
    imlist = mm.list()

    ap = multiprocessing.Process(target=ai.run, args=(imlist,))
    sp = multiprocessing.Process(target=server.run, args=(imlist,))

    ap.start()
    sp.start()
    ap.join()
    sp.terminate()