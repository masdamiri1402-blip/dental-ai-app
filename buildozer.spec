[app]

title = Dental AI
package.name = dentalai
package.domain = com.masoudamiri
version = 1.0
source.main = main.py
source.include_exts = py,png,jpg,jpeg,ttf,kv,pt,ptl
source.include_patterns = dental_model_mobile.ptl,labels.txt
source.dir = .
orientation = portrait
requirements = python3,kivy,torch,torchvision,pillow,numpy
android.api = 31
android.minapi = 21
android.ndk = 25b
android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.archs = arm64-v8a,armeabi-v7a
android.enable_androidx = True
fullname = Dental AI by Masoud Amiri
description = AI-powered dental diagnosis application
author = Masoud Amiri

[buildozer]

log_level = 2
warn_on_root = 1