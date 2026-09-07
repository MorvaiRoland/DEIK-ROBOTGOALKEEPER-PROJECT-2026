import time
import ximea.xiapi as xiapi

print("MASTER Kamera megnyitása méréshez...")
cam_master = xiapi.Camera(dev_id=1) # Jobb kamera a master
cam_master.open_device()

print("Kamera inicializálva.")
print("GPO PORT2 (Piros) és PORT3 (Barna) beállítása fix MAGAS (3.3V) szintre...")

try:
    cam_master.set_gpo_selector("XI_GPO_PORT2")
    cam_master.set_gpo_mode("XI_GPO_ON")
    
    cam_master.set_gpo_selector("XI_GPO_PORT3")
    cam_master.set_gpo_mode("XI_GPO_ON")
    
    print("\n" + "="*60)
    print(" SIKER! A Master Piros és Barna kábelén fix 3.3V-nak kell lennie.")
    print("="*60 + "\n")
except Exception as e:
    print(f"HIBA a GPO beállításakor: {e}")

print("A kamera 60 másodpercig nyitva marad a méréshez...")
for i in range(60, 0, -1):
    print(f"Hátralévő idő: {i} másodperc", end='\r')
    time.sleep(1)

cam_master.close_device()
print("\nKamera lezárva. Teszt vége.")
