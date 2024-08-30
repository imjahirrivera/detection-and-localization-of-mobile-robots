from ultralytics import YOLO
import cv2 as cv
import numpy as np
import pyzbar.pyzbar as pyzbar

drawing = False  # true if mouse is pressed
src_x, src_y = -1, -1
dst_x, dst_y = -1, -1

src_list = []
dst_list = [[226, 275], [326, 276], [326, 176], [226, 176]]  # Coordenadas iniciales
yolo_points = []  # Nueva lista para los puntos centrales de detección YOLO
yolo_active = False  # Flag para activar o desactivar YOLO
qr_saved = False  # Flag para verificar si los puntos del QR ya fueron guardados
temp_qr_points = []  # Lista temporal para almacenar los puntos del QR
merge_active = False  # Flag para activar la ventana de merge en tiempo real

# mouse callback function
def select_points_src(event, x, y, flags, param):
    global src_x, src_y, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        src_x, src_y = x, y
        cv.circle(frame, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

# mouse callback function
def select_points_dst(event, x, y, flags, param):
    global dst_x, dst_y, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        dst_x, dst_y = x, y
        cv.circle(dst_copy, (x, y), 5, (0, 0, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

def get_plan_view(src, dst):
    src_pts = np.array(src_list).reshape(-1, 1, 2)
    dst_pts = np.array(dst_list).reshape(-1, 1, 2)
    H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    #print("H:")
    #print(H)

    # Crear una imagen de fondo del mismo tamaño que dst, con un color blanco
    background = np.zeros((dst.shape[0], dst.shape[1], 4), dtype=np.uint8)

    plan_view = cv.warpPerspective(background, H, (dst.shape[1], dst.shape[0]))
    return plan_view, H

def merge_views(src, dst):
    plan_view, H = get_plan_view(src, dst)
    for i in range(0, dst.shape[0]):
        for j in range(0, dst.shape[1]):
            if (plan_view.item(i, j, 0) == 0 and
                plan_view.item(i, j, 1) == 0 and
                plan_view.item(i, j, 2) == 0):
                plan_view.itemset((i, j, 0), dst.item(i, j, 0))
                plan_view.itemset((i, j, 1), dst.item(i, j, 1))
                plan_view.itemset((i, j, 2), dst.item(i, j, 2))

    # Dibujar los puntos de YOLO en la vista combinada
    for point in yolo_points:
        # Transformar los puntos de YOLO a la vista combinada
        point_transformed = cv.perspectiveTransform(np.array([[point]], dtype=np.float32), H)[0][0]
        cv.circle(plan_view, (int(point_transformed[0]), int(point_transformed[1])), 5, (255, 0, 0), -1)  # Círculo rojo para puntos YOLO

    return plan_view

def decode_and_draw(im):
    global temp_qr_points
    decodedObjects = pyzbar.decode(im)
    temp_qr_points = []  # Limpiar la lista temporal antes de agregar nuevos puntos
    for obj in decodedObjects:
        points = obj.polygon
        if len(points) == 4:
            print('QR Coordinates:')
            for point in points:
                temp_qr_points.append([point.x, point.y])
                color = (0, 255, 0) if qr_saved else (0, 0, 255)
                cv.circle(im, (point.x, point.y), 5, color, -1)
                print(f'({point.x}, {point.y})')
            print('\n')

def draw_saved_qr_points(im):
    for point in src_list:
        cv.circle(im, (point[0], point[1]), 5, (0, 255, 0), -1)  # Círculo verde para puntos guardados

def add_yolo_detections(frame):
    global yolo_points  # Referenciar la variable global yolo_points
    yolo_points = []  # Limpiar la lista antes de agregar nuevos puntos

    resultados = model.predict(frame, imgsz=640)
    anotaciones = resultados[0].plot()

    for result in resultados[0].boxes:
        x_center = int((result.xyxy[0][0] + result.xyxy[0][2]) / 2)
        y_center = int((result.xyxy[0][1] + result.xyxy[0][3]) / 2)
        yolo_points.append([x_center, y_center])
        cv.circle(anotaciones, (x_center, y_center), 5, (255, 0, 0), -1)  # Centro rojo para detecciones YOLO

    return anotaciones

# Leer el modelo
model = YOLO("best.pt")

# Abrir el video
cap = cv.VideoCapture("Videos/video2.mp4")

# Leer la primera imagen de dst
dst = cv.imread('imgs/lab_homography.png', -1)
dst_copy = dst.copy()
dst_copy = cv.resize(dst_copy, (dst_copy.shape[1] // 2, dst_copy.shape[0] // 2))

# Dibujar los puntos iniciales en dst_copy
for point in dst_list:
    cv.circle(dst_copy, (point[0], point[1]), 5, (0, 255, 0), -1)

cv.namedWindow('src')
cv.setMouseCallback('src', select_points_src)

frame_counter = 0  # Contador de frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    if not yolo_active:
        # Detectar solo el QR
        decode_and_draw(frame)
        if qr_saved:
            draw_saved_qr_points(frame)
    else:
        # Procesar la imagen con YOLO
        anotaciones = add_yolo_detections(frame)
        frame = anotaciones
        draw_saved_qr_points(frame)

    # Si la ventana de merge está activa, actualizarla cada n frames
    if merge_active:
        if frame_counter % 10 == 0:  # Cambia 5 por un número mayor o menor para ajustar la frecuencia
            merge = merge_views(frame, dst_copy)
            cv.imshow("merge", merge)

    cv.imshow('src', frame)

    frame_counter += 1  # Incrementar el contador de frames

    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):
        yolo_active = True  # Activar YOLO y detener la detección de QR
        qr_saved = True  # Marcar que los puntos del QR están guardados
        src_list = temp_qr_points.copy()  # Guardar los puntos actuales del QR
        print("QR points saved. YOLO detection activated.")
    elif k == ord('m'):
        merge_active = not merge_active  # Alternar el estado de la ventana de merge
        if merge_active:
            print("Merge view activated.")
        else:
            print("Merge view deactivated.")
            cv.destroyWindow("merge")  # Cerrar la ventana de merge cuando se desactiva
    elif k == ord('p'):
        print(src_list)
    elif k == 27:  # Esc key
        break

cap.release()
cv.destroyAllWindows()