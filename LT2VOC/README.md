## Labeling Tag to VOC Format
### Description

* Labeling Tag 를 Pascal voc Format 으로 변경하는 기능을 제공
  * convert2VOC.py
    * openCV_EAST, YOLO, FACENET Format 을 voc format 으로 변경하는 기능 제공 
  

### 사용 방법
1. file_path 설정
  * convert2VOC.py 에서 변경을 원하는 폴더의 경로를 file_path 로 설정
  * 동일 경로에 .xml 파일이 저장됨 
  
2. Required Parameters
  * EAST mode (default 값)
```--op EAST 
```
  * YOLO mode
  ```--op YOLO 
```
  * FACENET mode
  ```--op FACENET 
```
3. Command line example
```python convert2VOC.py --op EAST 
```