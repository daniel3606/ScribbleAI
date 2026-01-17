import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image, ImageFilter  # FIX: import ImageFilter

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except Exception:
    st_canvas = None
    HAS_CANVAS = False

import random

# Ensure CUDA is available or fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Helpers for Pillow compatibility ----------
# FIX: Handle PIL resampling enum across versions
Resampling = getattr(Image, "Resampling", Image)

# ---------- Streamlit dialog compatibility ----------
DIALOG_DECORATOR = None
if hasattr(st, 'dialog'):
    DIALOG_DECORATOR = st.dialog
elif hasattr(st, 'experimental_dialog'):
    DIALOG_DECORATOR = st.experimental_dialog


# ---------- Model ----------
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ---------- Dataset ----------
class QuickDrawDataset(Dataset):
    def __init__(self, data, labels, transform=None, augment=False):
        self.data = torch.FloatTensor(data).reshape(-1, 1, 28, 28)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.labels[idx]

        if self.augment:
            # rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                img_pil = transforms.ToPILImage()(img)
                img_pil = transforms.functional.rotate(img_pil, angle, fill=0)
                img = transforms.ToTensor()(img_pil)
            # translation
            if random.random() > 0.5:
                max_shift = 3
                tx = random.randint(-max_shift, max_shift)
                ty = random.randint(-max_shift, max_shift)
                img_pil = transforms.ToPILImage()(img)
                img_pil = transforms.functional.affine(
                    img_pil, angle=0, translate=(tx, ty), scale=1.0, shear=0, fill=0
                )
                img = transforms.ToTensor()(img_pil)

        if self.transform:
            img = self.transform(img)

        return img, label

# ---------- Preprocessing ----------
def preprocess_drawing(img_array):
    """Normalize to [0,1]"""
    img_array = img_array.astype("float32") / 255.0
    img_array = np.clip(img_array, 0, 1)
    return img_array

def _rgba_to_rgb_on_white(pil_img):
    # FIX: Properly flatten alpha onto white background
    if pil_img.mode == "RGBA":
        bg = Image.new("RGB", pil_img.size, (255, 255, 255))
        bg.paste(pil_img, mask=pil_img.split()[-1])
        return bg
    if pil_img.mode == "LA":
        bg = Image.new("L", pil_img.size, 255)
        bg.paste(pil_img.convert("L"), mask=pil_img.split()[-1])
        return bg
    return pil_img.convert("RGB")

def preprocess_canvas_drawing(canvas_img):
    """
    Convert Streamlit canvas RGBA to 28x28 grayscale matching training: WHITE strokes on BLACK bg.
    """
    # To PIL
    if isinstance(canvas_img, np.ndarray):
        pil = Image.fromarray(canvas_img.astype("uint8"))
    else:
        pil = canvas_img

    # FIX: handle alpha correctly against white
    pil = _rgba_to_rgb_on_white(pil)

    # To grayscale
    pil = pil.convert("L")
    arr = np.array(pil)

    # Canvas usually has black strokes (0) on white bg (255).
    # We want white strokes on black bg like training. So invert ONCE.
    # FIX: single, deterministic invert to match training distribution
    arr = 255 - arr

    # Find bbox of foreground (now strokes are bright: high values)
    thresh = 15  # low threshold after inversion
    rows = np.any(arr > thresh, axis=1)
    cols = np.any(arr > thresh, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 10
        rmin = max(0, rmin - pad)
        rmax = min(arr.shape[0] - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(arr.shape[1] - 1, cmax + pad)
        arr = arr[rmin:rmax + 1, cmin:cmax + 1]

    # Resize to 28x28
    pil = Image.fromarray(arr)
    pil = pil.resize((28, 28), Resampling.LANCZOS)

    arr = np.array(pil).astype("float32") / 255.0

    # Optional gentle contrast bump (stay safe in [0,1])
    arr = np.clip((arr - 0.5) * 1.2 + 0.5, 0.0, 1.0)

    # NOTE: do NOT invert again — we already matched training (white on black)
    return arr


def preprocess_uploaded_image(pil_img):
    """
    Convert an uploaded image to 28x28 grayscale matching training: WHITE strokes on BLACK background.
    """
    pil = pil_img
    if not isinstance(pil, Image.Image):
        pil = Image.fromarray(np.array(pil_img).astype('uint8'))

    pil = pil.convert('L')
    arr = np.array(pil).astype('uint8')

    # Heuristic: if background is bright, invert so strokes become bright on dark
    if arr.mean() > 127:
        arr = 255 - arr

    # Crop around foreground
    thresh = 15
    rows = np.any(arr > thresh, axis=1)
    cols = np.any(arr > thresh, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = 10
        rmin = max(0, rmin - pad)
        rmax = min(arr.shape[0] - 1, rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(arr.shape[1] - 1, cmax + pad)
        arr = arr[rmin:rmax + 1, cmin:cmax + 1]

    pil2 = Image.fromarray(arr)
    pil2 = pil2.resize((28, 28), Resampling.LANCZOS)

    out = np.array(pil2).astype('float32') / 255.0
    out = np.clip((out - 0.5) * 1.2 + 0.5, 0.0, 1.0)
    return out

# ---------- Synthetic data ----------
def generate_synthetic_drawing(category, size=28):
    img = np.zeros((size, size), dtype=np.float32)

    offset_x = random.randint(-3, 3)
    offset_y = random.randint(-3, 3)
    scale = random.uniform(0.7, 1.0)

    if 'airplane' in category.lower():
        cy = size // 2 + offset_y
        cx = size // 2 + offset_x
        fus_len = int(size // 2 * scale)
        fus_w = max(2, int(3 * scale))
        img[cy - fus_w // 2: cy + fus_w // 2,
            max(0, cx - fus_len): min(size, cx + fus_len)] = random.uniform(0.7, 1.0)
        wing_w = int(size // 3 * scale)
        img[max(0, cy - wing_w): min(size, cy + wing_w), cx - 2: cx + 2] = random.uniform(0.7, 1.0)
        tail_h = int(size // 4 * scale)
        img[max(0, cy - tail_h): cy, max(0, cx + fus_len - 3): min(size, cx + fus_len)] = random.uniform(0.7, 1.0)

    elif 'banana' in category.lower():
        curve_type = random.choice(['sin', 'parabola', 'arc'])
        thickness = random.randint(2, 4)
        for i in range(size // 6, 5 * size // 6):
            if curve_type == 'sin':
                offset = int(size // 5 * np.sin((i - size // 6) / (size * 0.7) * np.pi))
            elif curve_type == 'parabola':
                offset = int((i - size // 2) ** 2 / (size * 2))
            else:
                offset = int(np.sqrt(max(0, (size // 3) ** 2 - (i - size // 2) ** 2)) - size // 6)
            x = i + offset_x
            y = size // 2 + offset + offset_y
            if 0 <= x < size:
                y0 = max(0, y - thickness)
                y1 = min(size, y + thickness)
                img[y0:y1, x] = random.uniform(0.8, 1.0)

    elif 'cat' in category.lower():
        cy = size // 2 + offset_y
        cx = size // 2 + offset_x
        r = int(size // 4 * scale)
        for i in range(max(0, cy - r), min(size, cy + r)):
            for j in range(max(0, cx - r), min(size, cx + r)):
                if (i - cy) ** 2 + (j - cx) ** 2 < r ** 2:
                    img[i, j] = random.uniform(0.7, 1.0)
        ear = int(r * 0.7)
        for i in range(ear):
            w = max(1, (ear - i) // 2)
            y = max(0, cy - r - i)
            if y < size:
                lx = max(0, cx - r // 2 - w)
                rx = min(size, cx + r // 2 + w)
                img[y, lx: min(size, lx + 2 * w)] = random.uniform(0.7, 1.0)
                img[y, max(0, rx - 2 * w): min(size, rx)] = random.uniform(0.7, 1.0)

    elif 'dog' in category.lower():
        cy = size // 2 + offset_y
        cx = size // 2 + offset_x
        r = int(size // 4 * scale)
        for i in range(max(0, cy - r), min(size, cy + r)):
            for j in range(max(0, cx - r), min(size, cx + r)):
                if (i - cy) ** 2 + (j - cx) ** 2 < r ** 2:
                    img[i, j] = random.uniform(0.7, 1.0)
        er = int(r * 0.5)
        lex, ley = cx - r, cy + r // 3
        rex, rey = cx + r, cy + r // 3
        for i in range(max(0, ley - er), min(size, ley + er)):
            for j in range(max(0, lex - er), min(size, lex + er)):
                if (i - ley) ** 2 + (j - lex) ** 2 < er ** 2:
                    img[i, j] = random.uniform(0.7, 1.0)
        for i in range(max(0, rey - er), min(size, rey + er)):
            for j in range(max(0, rex - er), min(size, rex + er)):
                if (i - rey) ** 2 + (j - rex) ** 2 < er ** 2:
                    img[i, j] = random.uniform(0.7, 1.0)

    elif 'car' in category.lower():
        cy = size // 2 + offset_y
        cx = size // 2 + offset_x
        bh = int(size // 4 * scale)
        bw = int(size // 2 * scale)
        img[max(0, cy - bh): min(size, cy + bh),
            max(0, cx - bw): min(size, cx + bw)] = random.uniform(0.7, 1.0)
        ch = int(size // 6 * scale)
        cw = int(size // 3 * scale)
        img[max(0, cy - bh - ch): max(0, cy - bh),
            max(0, cx - cw): min(size, cx + cw)] = random.uniform(0.7, 1.0)
        wr = int(3 * scale)
        wy = min(size - wr - 1, cy + bh)
        for i in range(-wr, wr + 1):
            for j in range(-wr, wr + 1):
                if i ** 2 + j ** 2 <= wr ** 2:
                    xL = cx - bw // 2
                    xR = cx + bw // 2
                    if 0 <= wy + i < size and 0 <= xL + j < size:
                        img[wy + i, xL + j] = random.uniform(0.8, 1.0)
                    if 0 <= wy + i < size and 0 <= xR + j < size:
                        img[wy + i, xR + j] = random.uniform(0.8, 1.0)

    elif 'tree' in category.lower():
        cy = size // 2 + offset_y
        cx = size // 2 + offset_x
        ftype = random.choice(['triangle', 'circle'])
        fs = int(size // 3 * scale)
        if ftype == 'triangle':
            for i in range(fs):
                w = int((fs - i) * 0.8)
                y = max(0, cy - fs // 2 + i)
                if y < size:
                    img[y, max(0, cx - w): min(size, cx + w)] = random.uniform(0.7, 1.0)
        else:
            for i in range(max(0, cy - fs), min(size, cy)):
                for j in range(max(0, cx - fs), min(size, cx + fs)):
                    if (i - (cy - fs // 2)) ** 2 + (j - cx) ** 2 < fs ** 2:
                        img[i, j] = random.uniform(0.7, 1.0)
        th = int(size // 3 * scale)
        tw = max(2, int(3 * scale))
        img[min(size - th, cy): min(size, cy + th),
            cx - tw // 2: cx + tw // 2] = random.uniform(0.6, 0.9)

    else:
        c = size // 2
        r = int(size // 3 * scale)
        for i in range(max(0, c - r), min(size, c + r)):
            for j in range(max(0, c - r), min(size, c + r)):
                if (i - c) ** 2 + (j - c) ** 2 < r ** 2:
                    img[i, j] = random.uniform(0.7, 1.0)

    # noise
    noise = np.random.rand(size, size) * random.uniform(0.05, 0.15)
    img = np.clip(img + noise, 0, 1)

    # random rotate
    if random.random() > 0.3:
        pil = Image.fromarray((img * 255).astype(np.uint8))
        pil = pil.rotate(random.uniform(-15, 15), fillcolor=0)
        img = np.array(pil).astype(np.float32) / 255.0

    # random slight blur
    if random.random() > 0.7:
        pil = Image.fromarray((img * 255).astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=1))  # FIX: use ImageFilter
        img = np.array(pil).astype(np.float32) / 255.0

    return img

def load_quickdraw_data(categories, samples_per_category=1000):
    if not categories:
        raise ValueError("No categories selected. Please select at least one category.")
    all_data, all_labels = [], []
    for label, category in enumerate(categories):
        print(f"Generating {category} dataset...")
        data = []
        for _ in range(samples_per_category):
            img = generate_synthetic_drawing(category)
            processed = preprocess_drawing(img * 255)
            data.append(processed.flatten())
        labels = np.full(len(data), label)
        all_data.append(data)
        all_labels.append(labels)
    all_data = np.concatenate(all_data)
    all_labels = np.concatenate(all_labels)
    print(f"Loaded {len(all_data)} samples.")
    return all_data, all_labels

# ---------- Training ----------
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, target_accuracy=95):
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 3
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / max(1, len(train_loader))
        epoch_accuracy = 100 * correct / max(1, total)

        model.eval()
        val_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                v_total += labels.size(0)
                v_correct += (preds == labels).sum().item()

        val_epoch_loss = val_loss / max(1, len(val_loader))
        val_epoch_accuracy = 100 * v_correct / max(1, v_total)

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")
        print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.2f}%")

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break

    if best_model is not None:
        model.load_state_dict(best_model)
    return model

# ---------- Prediction ----------
def predict_drawing(model, drawing, categories):
    model.eval()
    with torch.no_grad():
        if isinstance(drawing, np.ndarray) and drawing.ndim == 1:
            drawing = drawing.reshape(28, 28)
        processed = preprocess_drawing(drawing) if drawing.max() > 1 else drawing
        input_tensor = torch.FloatTensor(processed).reshape(1, 1, 28, 28).to(device)
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_probs, top_idx = torch.topk(probs, min(3, len(categories)))
        predictions = [(categories[i.item()], p.item()) for p, i in zip(top_probs, top_idx)]
        return predictions[0][0], predictions[0][1], predictions

# ---------- Prediction Modal ----------
if DIALOG_DECORATOR is not None:

    @DIALOG_DECORATOR("Prediction")
    def prediction_dialog():
        payload = st.session_state.get('pred_payload')
        if not payload:
            st.write('No prediction to show.')
            if st.button('Close'):
                st.rerun()
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write('Input')
            st.image(payload['raw'], width=200)
        with col2:
            st.write('Preprocessed')
            st.image(payload['processed'], width=200, clamp=True)
        with col3:
            st.write('Prediction')
            st.success(f"{payload['pred_class']}")
            st.metric('Confidence', f"{payload['confidence']*100:.1f}%")

        st.write('All predictions')
        for i, (cls, conf) in enumerate(payload['all_predictions'], 1):
            st.write(f"{i}. {cls}: {conf*100:.1f}%")
            st.progress(float(conf))

        if st.button('Close'):
            st.session_state['pred_payload'] = None
            st.rerun()


# ---------- Streamlit App ----------
def main():
    st.title("🎨 QuickDraw AI - Drawing Recognition")

    st.markdown("""
    ### Welcome! 
    1) Pick categories in the sidebar  
    2) Train the model  
    3) Draw below and predict!
    """)

    categories = st.sidebar.multiselect(
        "Select categories to train on",
        ['airplane', 'banana', 'cat', 'dog', 'car', 'tree'],
        default=['airplane', 'banana', 'cat'],
        help="Select 2-4 categories for best results"
    )

    if len(categories) == 0:
        st.sidebar.warning("⚠️ Select at least one category!")
    elif len(categories) == 1:
        st.sidebar.info("💡 Consider selecting 2+ categories for a better demo")
    else:
        st.sidebar.success(f"✓ {len(categories)} categories selected")

    samples_per_category = st.sidebar.slider("Samples per category", 500, 5000, 2000, 500)
    target_accuracy = st.sidebar.slider("Target Validation Accuracy (%)", 50, 100, 85, 5)
    num_epochs = st.sidebar.slider("Training Epochs", 5, 20, 12, 1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("Click **Train Model** to begin.")

    if st.sidebar.button("🚀 Train Model", type="primary"):
        if len(categories) == 0:
            st.error("❌ Please select at least one category!")
        else:
            try:
                with st.spinner("Training model..."):
                    all_data, all_labels = load_quickdraw_data(categories, samples_per_category)
                    X_train, X_val, y_train, y_val = train_test_split(
                        all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels
                    )
                    train_dataset = QuickDrawDataset(X_train, y_train, augment=True)
                    val_dataset = QuickDrawDataset(X_val, y_val, augment=False)

                    # FIX: num_workers=0 for Streamlit environments
                    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
                    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

                    model = CNNModel(num_classes=len(categories)).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=0.001)

                    model = train_model(model, train_loader, val_loader, criterion, optimizer,
                                        num_epochs=num_epochs, target_accuracy=target_accuracy)

                    st.session_state['model'] = model
                    st.session_state['categories'] = categories
                    st.session_state['sample_images'] = {cat: generate_synthetic_drawing(cat) for cat in categories}

                    st.success("✅ Model trained! Draw below.")
                    st.balloons()
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")

    if 'model' not in st.session_state:
        st.info("👆 Train the model in the sidebar to enable the canvas.")
        st.image("https://via.placeholder.com/300x300.png?text=Canvas+appears+after+training",
                 caption="Drawing canvas preview", width=300)

    if 'model' in st.session_state:
        if 'sample_images' in st.session_state and st.session_state['sample_images']:
            with st.expander("📚 Example training images"):
                cols = st.columns(len(st.session_state['categories']))
                for i, cat in enumerate(st.session_state['categories']):
                    with cols[i]:
                        st.write(f"**{cat}**")
                        st.image(st.session_state['sample_images'][cat], width=100, clamp=True)

        st.markdown("---")
        st.subheader("Upload an image")
        uploaded = st.file_uploader("Upload a doodle image (png, jpg)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            pil = Image.open(uploaded)
            processed_upl = preprocess_uploaded_image(pil)
            col_u1, col_u2 = st.columns(2)
            with col_u1:
                st.write('Uploaded')
                st.image(pil, width=200)
            with col_u2:
                st.write('Preprocessed')
                st.image(processed_upl, width=200, clamp=True)

            if st.button("🎯 Predict Uploaded Image", type="primary", use_container_width=True):
                pred_class, confidence, all_predictions = predict_drawing(
                    st.session_state['model'], processed_upl, st.session_state['categories']
                )

                st.session_state['pred_payload'] = {
                    'raw': pil,
                    'processed': processed_upl,
                    'pred_class': pred_class,
                    'confidence': confidence,
                    'all_predictions': all_predictions,
                }

                if DIALOG_DECORATOR is not None:
                    prediction_dialog()
                else:
                    st.subheader('Prediction')
                    st.write(pred_class)
                    st.write(f"Confidence: {confidence*100:.1f}%")
                    for cls, conf in all_predictions:
                        st.write(f"{cls}: {conf*100:.1f}%")
                        st.progress(float(conf))

        st.markdown("---")
        st.subheader("✏️ Draw Your Doodle")
        st.write(f"Draw one of: **{', '.join(st.session_state['categories'])}**")

        col_draw1, col_draw2, col_draw3 = st.columns([2, 1, 1])
        with col_draw2:
            if st.button("🗑️ Clear Canvas", use_container_width=True):
                st.session_state['canvas_key'] = st.session_state.get('canvas_key', 0) + 1
                st.rerun()
        with col_draw3:
            if st.button("🔄 Reset Model", use_container_width=True):
                for k in ['model', 'categories', 'sample_images']:
                    if k in st.session_state:
                        del st.session_state[k]
                st.rerun()

        if 'canvas_key' not in st.session_state:
            st.session_state['canvas_key'] = 0

        if not HAS_CANVAS:
            st.info('Canvas requires streamlit-drawable-canvas. Install it to draw, or use Upload an image above.')
            canvas_result = None
        else:
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=8,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=400,
                width=400,
                drawing_mode="freedraw",
                key=f"canvas_{st.session_state['canvas_key']}",
            )

        if canvas_result is not None and canvas_result.image_data is not None:
            # any drawing?
            canvas_sum = np.sum(canvas_result.image_data[:, :, :3])
            has_drawing = canvas_sum < (255 * 3 * canvas_result.image_data.shape[0] * canvas_result.image_data.shape[1])  # not pure white

            if has_drawing and st.button("🎯 Predict Drawing", type="primary", use_container_width=True):
                processed_img = preprocess_canvas_drawing(canvas_result.image_data)
                pred_class, confidence, all_predictions = predict_drawing(
                    st.session_state['model'], processed_img, st.session_state['categories']
                )

                st.session_state['pred_payload'] = {
                    'raw': canvas_result.image_data,
                    'processed': processed_img,
                    'pred_class': pred_class,
                    'confidence': confidence,
                    'all_predictions': all_predictions,
                }

                if DIALOG_DECORATOR is not None:
                    prediction_dialog()
                else:
                    st.subheader('Prediction')
                    st.write(pred_class)
                    st.write(f"Confidence: {confidence*100:.1f}%")
                    for cls, conf in all_predictions:
                        st.write(f"{cls}: {conf*100:.1f}%")
                        st.progress(float(conf))
            elif not has_drawing:
                st.info("👆 Draw something and then click **Predict Drawing**")

if __name__ == "__main__":
    main()
