# 1_stitcher_basico.py
# uso: python 1_stitcher_basico.py --imgs a.jpg b.jpg c.jpg --modo panorama
import cv2, argparse, sys
p=argparse.ArgumentParser()
p.add_argument("--imgs", nargs="+", required=True)
p.add_argument("--modo", choices=["panorama","scans"], default="panorama")
p.add_argument("--try_gpu", action="store_true")
a=p.parse_args()
mode=cv2.Stitcher_PANORAMA if a.modo=="panorama" else cv2.Stitcher_SCANS
st=cv2.Stitcher_create(mode)
st.setPanoConfidenceThresh(0.5)
st.setWarper(cv2.PyRotationWarper("spherical", 1.0))  # bom p/ panoramas
status, pano = st.stitch([cv2.imread(x) for x in a.imgs])
if status!=cv2.STITCHER_OK: sys.exit(f"falhou com status {status}")
cv2.imwrite("pano_stitcher.jpg", pano)
print("ok: pano_stitcher.jpg")
