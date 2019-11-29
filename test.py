import cv2
import numpy as np
import prediction
import multiprocessing
from goprocam import GoProCamera
from goprocam import constants


#problem the detection frame is still too far behind, and when its taking priority, its offsetting the ponits too much. how to calculate which one is more accurate one

def determine_bound(nose,reye, leye):
    try:
        rightLine = leftLine = (leye[0] + reye[0]) / 2
        topLine = bottomLine = (min(leye[1], reye[1]) + nose[1])/2
        return (int(leftLine), int(rightLine), int(topLine),int(bottomLine))
    except Exception as e:
        return None


def run_detection(recieve_que, send_que, after_frame_que):
    lk_params = dict(winSize = (15, 15),
                     maxLevel = 6,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
     
    detection = prediction.OpenPose('mobile.pb', convert_csv=False)
    while True:
        if not recieve_que.empty():
            frame, frame_number = recieve_que.get()
            result = detection.detect(frame)
            center = detection.draw_humans(frame, result, imgcopy=False)
            

                
            try:
                if center.any():
                    '''
                    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    while not after_frame_que.empty():
                        new_gray, num = after_frame_que.get()
                        if (num - frame_number) > 9:
                            break
                        old_points = center
                        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_points, None, **lk_params)
                        old_points = new_points
                        old_gray = new_gray
                    '''
                    send_que.put((center, frame_number))
            except AttributeError as e:
                send_que.put((np.array([[],[],[]]), frame_number))


def checkCommand(bound, nose, reye, leye, frame):

    try:

        if reye[1] > bound[3] or leye[1] > bound[3]:
            return "Breaking!!"
        if nose[1] < bound[2]:
            return "Going Forward"
        if reye[0] > bound[0]:
            return "Turning Left"
        if leye[0] < bound[1]:
            return "Turning Right"
    except Exception as e:
        return "BOUND"
    return None


def main():
        
    #cap = cv2.VideoCapture("udp://127.0.0.1:10000")
    cap = cv2.VideoCapture('./test.mp4')
    # Create old frame
    _, frame = cap.read()
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
    # Lucas kanade params
    lk_params = dict(winSize = (25, 25),
                     maxLevel = 4,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
     
    # Mouse function

    bound = None
    point_selected = False
    stop_detection = False
    first_time=True
    old_points = np.array([[]])
    frame_num = 1
    send_que = multiprocessing.Queue()
    recieve_que = multiprocessing.Queue()
    after_frame_que = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_detection, args=(recieve_que, send_que, after_frame_que), daemon=True)
    p.start()

    while True:
        _, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if point_selected is False and stop_detection is False: #initial running the detection
            recieve_que.put((frame, frame_num))
            stop_detection = True
            stopped_frame = frame_num + 1

        #if stop_detection:
        #    if (frame_num - stopped_frame) %2 == 0 and (frame_num - stopped_frame) < 10:
        #       after_frame_que.put((gray_frame, frame_num))



            
        if not send_que.empty(): #gettign result back from detection
            center, frame_number = send_que.get()
            point_selected = True
            old_points = center

            print(frame_number)

            if frame_number <= 500 and frame_number > 1:
                bound = determine_bound(tuple(center[0]), tuple(center[1]), tuple(center[2]))
                inital_bound = bound
                first_time = False
            if frame_number > 500:
                command = checkCommand(bound, tuple(center[0]), tuple(center[1]), tuple(center[2]), frame)
                if command:
                    if command != "BOUND":
                        print(command)
                        bound = determine_bound(tuple(center[0]), tuple(center[1]), tuple(center[2]))
                    else:
                        bound = inital_bound

        if point_selected:
            if frame_num % 5 ==0:
                point_selected = False
                stop_detection = False
            new_points = center

        else:
            try:
                new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
            except Exception as e:
                pass

        try:
            old_gray = gray_frame.copy()
            old_points = new_points
            x1, y1,x2,y2,x3,y3 = new_points.ravel()

            cv2.circle(frame, (x1, y1), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 5, (0, 255, 0), -1)
            cv2.circle(frame, (x3, y3), 5, (0, 255, 0), -1)
        except Exception as e:
            pass
            
        if bound:
            height, width, _ = frame.shape
            cv2.line(frame, (bound[0], 0), (bound[0], height),(0, 255, 0), 1)
            cv2.line(frame, (bound[1], 0),(bound[1], height), (0, 255, 0), 1)
            cv2.line(frame, (0,bound[2]),(width, bound[2]), (0, 255, 0), 1)
            cv2.line(frame, (0, bound[3]),(width, bound[3]), (0, 255, 0), 1)
     
        cv2.imshow("Frame", frame)

        frame_num += 1
     
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()