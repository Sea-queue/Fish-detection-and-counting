from ultralytics import YOLO

weights = [
  "model/train49.pt",
  "model/train95.pt",
  "model/train110.pt",
  "model/train111.pt",
  "model/train113.pt",
]

for w in weights:
    m = YOLO(w)
    names = m.names
    print("\n", w)
    print("type(names):", type(names))
    print("names:", names)
    if isinstance(names, dict):
        print("max key:", max(names.keys()))
