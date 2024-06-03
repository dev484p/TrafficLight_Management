import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import matplotlib.pyplot as plt

def give_count(videopath,xl1=282,xl2=1004,y=308):
    model=YOLO('yolov9c.pt')
    class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    tracker=Tracker()

    # Coordinates of Line
    # xl1=282
    # xl2=1004
    # y=308

    cap=cv2.VideoCapture(videopath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('CountCars.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (frame_width, frame_height))
    down={}
    down_car={}
    down_bike={}
    down_truck={}
    down_bus={}

    counter_down=set()
    car_down=set()
    bike_down=set()
    truck_down=set()
    bus_down=set()

    while True:    
        ret,frame = cap.read()
        if not ret:
            break
        results=model.predict(frame)

        a=results[0].boxes.data
        a = a.detach().cpu().numpy()  
        px=pd.DataFrame(a).astype("float")
        list=[]
                
        for index,row in px.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])
            c=class_list[d]
            if ('car' in c) or ('motorcycle'in c) or ('bus' in c) or('truck' in c):
                list.append([x1,y1,x2,y2,c])

        bbox_id=tracker.update(list)
        for bbox in bbox_id:
            x3,y3,x4,y4,id,cl=bbox
            cx=int(x3+x4)//2
            cy=int(y3+y4)//2
            offset = 7
            
            if y < (cy + offset) and y > (cy - offset):
                down[id]=cy            #cy is current position. saving the ids of the cars which are touching the trigger line first. 
                if('car' in cl):
                    down_car[id]=cy
                elif('motorcycle'in cl):
                    down_bike[id]=cy
                elif('bus'in cl):
                    down_bus[id]=cy
                elif('truck'in cl):
                    down_truck[id]=cy
                    
                #This will tell us the travelling direction of the car.
                if id in down:         
                    cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    counter_down.add(id)
                if id in down_car:
                    cv2.putText(frame,("Car"),(cx,cy),cv2.FONT_HERSHEY_DUPLEX, 1, text_color, 1, cv2.LINE_AA) 
                    car_down.add(id)
                elif id in down_bike:
                    cv2.putText(frame,("Bike"),(cx,cy),cv2.FONT_HERSHEY_DUPLEX, 1, text_color, 1, cv2.LINE_AA) 
                    bike_down.add(id)
                elif id in down_truck:
                    cv2.putText(frame,("Truck"),(cx,cy),cv2.FONT_HERSHEY_DUPLEX, 1, text_color, 1, cv2.LINE_AA) 
                    truck_down.add(id)
                elif id in down_bus:
                    cv2.putText(frame,("Bus"),(cx,cy),cv2.FONT_HERSHEY_DUPLEX, 1, text_color, 1, cv2.LINE_AA) 
                    bus_down.add(id)

        # line
        text_color = (255,255,255)  # white color for text
        red_color = (255, 255,0)  # (B, G, R)   
        
        # print(down)
        cv2.line(frame,(xl1,y),(xl2,y),red_color,3)  #  starting cordinates and end of line cordinates
        cv2.putText(frame,('Trigger line'),(280,y),cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color, 1, cv2.LINE_AA) 

        
        downwards = (len(counter_down))
        downwards_car = (len(car_down))
        downwards_bike = (len(bike_down))
        downwards_bus = (len(bus_down))
        downwards_truck = (len(truck_down))
        
        cv2.putText(frame,('total going down - ')+ str(downwards),(60,40),cv2.FONT_HERSHEY_DUPLEX, 0.5, red_color, 1, cv2.LINE_AA) 
        cv2.putText(frame,('car down - ')+ str(downwards_car),(60,60),cv2.FONT_HERSHEY_DUPLEX, 0.5, red_color, 1, cv2.LINE_AA) 
        cv2.putText(frame,('bike down - ')+ str(downwards_bike),(60,80),cv2.FONT_HERSHEY_DUPLEX, 0.5, red_color, 1, cv2.LINE_AA)
        cv2.putText(frame,('bus down - ')+ str(downwards_bus),(60,100),cv2.FONT_HERSHEY_DUPLEX, 0.5, red_color, 1, cv2.LINE_AA) 
        cv2.putText(frame,('truck down - ')+ str(downwards_truck),(60,120),cv2.FONT_HERSHEY_DUPLEX, 0.5, red_color, 1, cv2.LINE_AA)
        
        out.write(frame)
    out.release()
    cap.release()
    opt_list=[len(car_down),len(bike_down),len(bus_down),len(truck_down)]
    return opt_list
    