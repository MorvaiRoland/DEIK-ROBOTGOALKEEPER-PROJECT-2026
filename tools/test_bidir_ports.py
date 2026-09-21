import time
import ximea.xiapi as xiapi

print("Kamerák inicializálása GPIO teszthez...")

cam_slave = xiapi.Camera(dev_id=0)  # Bal
cam_master = xiapi.Camera(dev_id=1) # Jobb

cam_slave.open_device()
cam_master.open_device()

print("Kamerák sikeresen megnyitva.")

def test_connection(cam_out, cam_in, out_port, in_port, label):
    print(f"\n--- Teszt: {label} ({out_port} -> {in_port}) ---")
    try:
        # Bemenet konfigurálása
        cam_in.set_gpo_selector(in_port.replace("GPI", "GPO"))
        cam_in.set_gpo_mode("XI_GPO_HIGH_IMPEDANCE")
        cam_in.set_gpi_selector(in_port)
        
        # Kimenet LOW
        cam_out.set_gpo_selector(out_port)
        cam_out.set_gpo_mode("XI_GPO_OFF") # LOW
        time.sleep(0.1)
        level_low = cam_in.get_gpi_level()
        
        # Kimenet HIGH
        cam_out.set_gpo_mode("XI_GPO_ON") # HIGH
        time.sleep(0.1)
        level_high = cam_in.get_gpi_level()
        
        # Vissza LOW
        cam_out.set_gpo_mode("XI_GPO_OFF")
        
        print(f"Eredmény: LOW jel = {level_low} | HIGH jel = {level_high}")
        if level_low == 0 and level_high == 1:
            print(">>> SIKERES KAPCSOLAT! A jel átment. <<<")
            return True
        else:
            print(">>> HIBA: A jel nem ment át. <<<")
            return False
            
    except Exception as e:
        print(f"Hiba a teszt során: {e}")
        return False

# PORT2 (Piros) és PORT3 (Barna) tesztelése mindkét irányban
test_connection(cam_master, cam_slave, "XI_GPO_PORT2", "XI_GPI_PORT2", "MASTER -> SLAVE (Piros kábel / PORT2)")
test_connection(cam_slave, cam_master, "XI_GPO_PORT2", "XI_GPI_PORT2", "SLAVE -> MASTER (Piros kábel / PORT2)")

test_connection(cam_master, cam_slave, "XI_GPO_PORT3", "XI_GPI_PORT3", "MASTER -> SLAVE (Barna kábel / PORT3)")
test_connection(cam_slave, cam_master, "XI_GPO_PORT3", "XI_GPI_PORT3", "SLAVE -> MASTER (Barna kábel / PORT3)")

cam_slave.close_device()
cam_master.close_device()
print("\nTeszt befejezve.")
