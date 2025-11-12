"""
Real-Time Cursor Control Using Hand Gestures with Vision Transformer
A complete implementation combining MediaPipe and Vision Transformer
"""
"""
# gesture_control.py - Main application

1. Real-time gesture recognition
2. Vision Transformer implementation
3. MediaPipe integration
4. Mouse control via PyAutoGUI
5. Data collection mode
"""

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time
from collections import deque
import math

# Disable PyAutoGUI failsafe for smooth operation
pyautogui.FAILSAFE = False

# ==================== VISION TRANSFORMER MODEL ====================

class PatchEmbedding(nn.Module):
    """Convert image into patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                             kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for gesture classification"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, 
                 n_classes=7, embed_dim=768, depth=12, n_heads=12, 
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 
                                         in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable parameters
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        return x


# ==================== GESTURE CONTROLLER ====================

class GestureController:
    """Main controller for hand gesture recognition and mouse control"""
    
    # Gesture definitions
    GESTURES = {
        0: "NONE",
        1: "MOVE",      # Index finger up - Move cursor
        2: "CLICK",     # Index + Middle up close - Left click
        3: "RIGHT_CLICK", # Index + Middle + Ring up - Right click
        4: "DRAG",      # Fist closed - Drag
        5: "SCROLL_UP", # Thumb up
        6: "SCROLL_DOWN" # Thumb down
    }
    
    def __init__(self, use_vit=True, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.use_vit = use_vit
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize Vision Transformer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.use_vit:
            self.model = VisionTransformer(
                img_size=224,
                patch_size=16,
                n_classes=7,
                embed_dim=384,  # Smaller for faster inference
                depth=6,        # Fewer layers for speed
                n_heads=6
            ).to(self.device)
            self.model.eval()
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Smoothing and state management
        self.prev_x, self.prev_y = 0, 0
        self.smooth_factor = 0.3
        self.gesture_buffer = deque(maxlen=5)
        self.click_cooldown = 0
        self.drag_mode = False
        self.last_gesture = 0
        
        # Frame region for hand detection
        self.frame_reduction = 100
        
    def get_hand_landmarks(self, frame):
        """Extract hand landmarks using MediaPipe"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        return results
    
    def extract_hand_roi(self, frame, landmarks, margin=40):
        """Extract region of interest around hand"""
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in landmarks.landmark]
        y_coords = [lm.y * h for lm in landmarks.landmark]
        
        x_min = max(0, int(min(x_coords)) - margin)
        x_max = min(w, int(max(x_coords)) + margin)
        y_min = max(0, int(min(y_coords)) - margin)
        y_max = min(h, int(max(y_coords)) + margin)
        
        roi = frame[y_min:y_max, x_min:x_max]
        return roi, (x_min, y_min, x_max, y_max)
    
    def recognize_gesture_vit(self, roi):
        """Recognize gesture using Vision Transformer"""
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return 0
        
        try:
            pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(img_tensor)
                _, predicted = torch.max(outputs, 1)
                gesture_id = predicted.item()
                
            return gesture_id
        except Exception as e:
            print(f"ViT prediction error: {e}")
            return 0
    
    def recognize_gesture_heuristic(self, landmarks):
        """Fallback: Recognize gesture using hand landmark heuristics"""
        fingers = []
        
        # Thumb
        if landmarks.landmark[4].x < landmarks.landmark[3].x:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers
        finger_tips = [8, 12, 16, 20]
        finger_pips = [6, 10, 14, 18]
        
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks.landmark[tip].y < landmarks.landmark[pip].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        # Gesture recognition logic
        if fingers == [0, 1, 0, 0, 0]:  # Only index
            return 1  # MOVE
        elif fingers == [0, 1, 1, 0, 0]:  # Index + Middle
            # Check if close together for click
            tip8 = landmarks.landmark[8]
            tip12 = landmarks.landmark[12]
            distance = math.sqrt((tip8.x - tip12.x)**2 + (tip8.y - tip12.y)**2)
            if distance < 0.05:
                return 2  # CLICK
            return 1  # MOVE
        elif fingers == [0, 1, 1, 1, 0]:  # Index + Middle + Ring
            return 3  # RIGHT_CLICK
        elif sum(fingers) == 0:  # Fist
            return 4  # DRAG
        elif fingers[0] == 1 and sum(fingers[1:]) == 0:  # Thumb up
            return 5  # SCROLL_UP
        elif fingers[0] == 1 and sum(fingers[1:]) == 4:  # All fingers up
            return 6  # SCROLL_DOWN
        
        return 0  # NONE
    
    def smooth_gesture(self, gesture):
        """Smooth gesture recognition using buffer"""
        self.gesture_buffer.append(gesture)
        if len(self.gesture_buffer) >= 3:
            # Majority voting
            return max(set(self.gesture_buffer), key=self.gesture_buffer.count)
        return gesture
    
    def move_cursor(self, landmarks, frame_shape):
        """Move cursor based on index finger position"""
        h, w, _ = frame_shape
        index_tip = landmarks.landmark[8]
        
        # Convert to screen coordinates
        x = int(index_tip.x * self.screen_width)
        y = int(index_tip.y * self.screen_height)
        
        # Smooth movement
        x = int(self.prev_x + (x - self.prev_x) * self.smooth_factor)
        y = int(self.prev_y + (y - self.prev_y) * self.smooth_factor)
        
        self.prev_x, self.prev_y = x, y
        pyautogui.moveTo(x, y, duration=0)
    
    def perform_action(self, gesture, landmarks, frame_shape):
        """Perform mouse action based on gesture"""
        gesture_name = self.GESTURES[gesture]
        
        if gesture == 1:  # MOVE
            self.move_cursor(landmarks, frame_shape)
            self.drag_mode = False
            
        elif gesture == 2:  # CLICK
            if self.click_cooldown == 0:
                pyautogui.click()
                self.click_cooldown = 15  # Cooldown frames
            self.drag_mode = False
            
        elif gesture == 3:  # RIGHT_CLICK
            if self.click_cooldown == 0:
                pyautogui.rightClick()
                self.click_cooldown = 15
            self.drag_mode = False
            
        elif gesture == 4:  # DRAG
            if not self.drag_mode:
                pyautogui.mouseDown()
                self.drag_mode = True
            self.move_cursor(landmarks, frame_shape)
            
        elif gesture == 5:  # SCROLL_UP
            pyautogui.scroll(20)
            
        elif gesture == 6:  # SCROLL_DOWN
            pyautogui.scroll(-20)
        
        # Handle drag release
        if self.drag_mode and gesture != 4:
            pyautogui.mouseUp()
            self.drag_mode = False
        
        # Decrease cooldown
        if self.click_cooldown > 0:
            self.click_cooldown -= 1
        
        self.last_gesture = gesture
        return gesture_name
    
    def run(self):
        """Main loop for gesture control"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n=== Vision Transformer Hand Gesture Control ===")
        print("Gestures:")
        print("  - Index finger up: Move cursor")
        print("  - Index + Middle close: Left click")
        print("  - Index + Middle + Ring: Right click")
        print("  - Fist: Drag")
        print("  - Thumb up: Scroll up")
        print("  - All fingers: Scroll down")
        print("\nPress 'q' to quit\n")
        
        fps_time = time.time()
        fps_counter = 0
        fps = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            # Get hand landmarks
            results = self.get_hand_landmarks(frame)
            
            gesture_name = "NONE"
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Recognize gesture
                    if self.use_vit:
                        roi, bbox = self.extract_hand_roi(frame, hand_landmarks)
                        gesture = self.recognize_gesture_vit(roi)
                        # Fallback to heuristic if ViT fails
                        if gesture == 0:
                            gesture = self.recognize_gesture_heuristic(hand_landmarks)
                    else:
                        gesture = self.recognize_gesture_heuristic(hand_landmarks)
                    
                    # Smooth gesture
                    gesture = self.smooth_gesture(gesture)
                    
                    # Perform action
                    gesture_name = self.perform_action(gesture, hand_landmarks, (h, w, 3))
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time > 1:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
            
            # Display info
            cv2.putText(frame, f"FPS: {fps}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Gesture: {gesture_name}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Mode: {'ViT' if self.use_vit else 'Heuristic'}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Hand Gesture Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


# ==================== TRAINING UTILITIES ====================

def create_training_dataset():
    """Utility to capture training data for gestures"""
    print("\n=== Gesture Training Data Collector ===")
    print("Press keys to capture gestures:")
    print("  0: NONE, 1: MOVE, 2: CLICK, 3: RIGHT_CLICK")
    print("  4: DRAG, 5: SCROLL_UP, 6: SCROLL_DOWN")
    print("  's': Save, 'q': Quit")
    
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    
    gesture_data = {i: [] for i in range(7)}
    current_gesture = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, 
                                      mp_hands.HAND_CONNECTIONS)
        
        if current_gesture is not None:
            cv2.putText(frame, f"Capturing: {current_gesture}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Training Data Collector', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('0') <= key <= ord('6'):
            current_gesture = key - ord('0')
            if results.multi_hand_landmarks:
                gesture_data[current_gesture].append(frame.copy())
                print(f"Captured gesture {current_gesture} - "
                      f"Total: {len(gesture_data[current_gesture])}")
        elif key == ord('s'):
            print("\nSaving dataset...")
            import os
            os.makedirs('gesture_dataset', exist_ok=True)
            for gesture_id, images in gesture_data.items():
                gesture_dir = f'gesture_dataset/gesture_{gesture_id}'
                os.makedirs(gesture_dir, exist_ok=True)
                for i, img in enumerate(images):
                    cv2.imwrite(f'{gesture_dir}/img_{i}.jpg', img)
            print("Dataset saved!")
    
    cap.release()
    cv2.destroyAllWindows()


# ==================== MAIN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Vision Transformer Hand Gesture Control')
    parser.add_argument('--mode', type=str, default='run', 
                       choices=['run', 'collect'],
                       help='Mode: run controller or collect training data')
    parser.add_argument('--use-vit', action='store_true', default=False,
                       help='Use Vision Transformer (requires trained model)')
    parser.add_argument('--screen-width', type=int, default=1920,
                       help='Screen width')
    parser.add_argument('--screen-height', type=int, default=1080,
                       help='Screen height')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        create_training_dataset()
    else:
        controller = GestureController(
            use_vit=args.use_vit,
            screen_width=args.screen_width,
            screen_height=args.screen_height
        )
        controller.run()
