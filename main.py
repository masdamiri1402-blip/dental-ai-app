"""
Dental AI Mobile Application
Developed by: Masoud Amiri
Framework: Kivy + PyTorch Mobile
"""

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
import torch
from PIL import Image
import torchvision.transforms as transforms
import os

# تنظیم اندازه پنجره برای تست
Window.size = (400, 700)

class DentalAIApp(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        
        # مسیر مدل
        self.model_path = 'dental_model_mobile.ptl'
        self.classes = ['Decay', 'Denervation', 'Healthy']
        
        # بارگذاری مدل
        self.load_model()
        
        # ساخت رابط کاربری
        self.build_ui()
    
    def load_model(self):
        """بارگذاری مدل PyTorch"""
        try:
            if os.path.exists(self.model_path):
                self.model = torch.jit.load(self.model_path, map_location='cpu')
                self.model.eval()
                print("Model loaded successfully!")
            else:
                self.model = None
                print(f"Model not found at {self.model_path}")
        except Exception as e:
            self.model = None
            print(f"Error loading model: {e}")
    
    def build_ui(self):
        """ساخت رابط کاربری"""
        
        # رنگ پس‌زمینه
        with self.canvas.before:
            Color(0.95, 0.95, 0.97, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)
        
        # عنوان
        header = BoxLayout(orientation='vertical', size_hint_y=0.2, padding=10, spacing=5)
        title = Label(
            text='[b]🦷 Dental AI[/b]',
            markup=True,
            font_size='28sp',
            color=(0.2, 0.4, 0.8, 1)
        )
        subtitle = Label(
            text='Developed by Masoud Amiri',
            font_size='14sp',
            color=(0.5, 0.5, 0.5, 1)
        )
        header.add_widget(title)
        header.add_widget(subtitle)
        self.add_widget(header)
        
        # نمایش تصویر
        self.image_display = KivyImage(
            source='',
            size_hint_y=0.4,
            allow_stretch=True,
            keep_ratio=True
        )
        self.add_widget(self.image_display)
        
        # دکمه‌ها
        buttons_layout = BoxLayout(orientation='vertical', size_hint_y=0.2, padding=20, spacing=15)
        
        # دکمه انتخاب عکس
        btn_choose = Button(
            text='📁 انتخاب عکس از گالری',
            font_size='18sp',
            background_color=(0.3, 0.6, 0.9, 1),
            color=(1, 1, 1, 1),
            bold=True
        )
        btn_choose.bind(on_press=self.choose_image)
        buttons_layout.add_widget(btn_choose)
        
        self.add_widget(buttons_layout)
        
        # نمایش نتایج
        self.result_label = Label(
            text='لطفا یک عکس رادیوگرافی دندان انتخاب کنید',
            font_size='16sp',
            size_hint_y=0.2,
            color=(0.3, 0.3, 0.3, 1),
            halign='center'
        )
        self.add_widget(self.result_label)
    
    def _update_rect(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
    
    def choose_image(self, instance):
        """انتخاب عکس از گالری"""
        content = BoxLayout(orientation='vertical')
        
        filechooser = FileChooserIconView(
            filters=['*.png', '*.jpg', '*.jpeg'],
            path=os.path.expanduser('~')
        )
        
        button_layout = BoxLayout(size_hint_y=0.1, spacing=10, padding=10)
        
        btn_select = Button(text='انتخاب', background_color=(0.3, 0.8, 0.3, 1))
        btn_cancel = Button(text='لغو', background_color=(0.8, 0.3, 0.3, 1))
        
        button_layout.add_widget(btn_select)
        button_layout.add_widget(btn_cancel)
        
        content.add_widget(filechooser)
        content.add_widget(button_layout)
        
        popup = Popup(
            title='انتخاب عکس',
            content=content,
            size_hint=(0.9, 0.9)
        )
        
        def select_file(instance):
            if filechooser.selection:
                self.process_image(filechooser.selection[0])
            popup.dismiss()
        
        btn_select.bind(on_press=select_file)
        btn_cancel.bind(on_press=popup.dismiss)
        
        popup.open()
    
    def process_image(self, image_path):
        """پردازش و تشخیص تصویر"""
        try:
            # نمایش تصویر
            self.image_display.source = image_path
            
            if self.model is None:
                self.result_label.text = '❌ خطا: مدل بارگذاری نشده است'
                return
            
            # پیش‌پردازش تصویر
            image = Image.open(image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0)
            
            # پیش‌بینی
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # نمایش نتایج
            results = []
            for i, cls in enumerate(self.classes):
                prob = probabilities[i].item() * 100
                emoji = '✅' if i == 2 else ('⚠️' if i == 0 else '❌')
                results.append(f"{emoji} {cls}: {prob:.1f}%")
            
            # پیدا کردن بالاترین احتمال
            max_idx = probabilities.argmax().item()
            prediction = self.classes[max_idx]
            confidence = probabilities[max_idx].item() * 100
            
            result_text = f"[b]تشخیص: {prediction} ({confidence:.1f}%)[/b]\n\n"
            result_text += "\n".join(results)
            
            self.result_label.text = result_text
            self.result_label.markup = True
            
        except Exception as e:
            self.result_label.text = f'❌ خطا در پردازش: {str(e)}'
            print(f"Error: {e}")

class DentalAI(App):
    def build(self):
        self.title = 'Dental AI'
        return DentalAIApp()

if __name__ == '__main__':
    DentalAI().run()
