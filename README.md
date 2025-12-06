# AI-WRTS System

Hệ thống quản lý học viên AI Writing & Reading Training System.

## 🚀 Hướng dẫn Setup Đầy đủ

### BƯỚC 1: TẠO DATABASE
```bash
mysql -u root -p
CREATE DATABASE AI_WRTS;
exit;
```

### BƯỚC 2: MIGRATE DATABASE
```bash
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

### BƯỚC 3: SEED ROLES (QUAN TRỌNG!)
```bash
python database/seed_roles.py
```

**Phải thấy:**
```
✅ Đã seed 4 roles thành công!
   - Student (STUDENT)
   - Instructor (INSTRUCTOR)
   - Administrator (ADMIN)
   - Manager (MANAGER)
```

### BƯỚC 4: SEED ADMIN (OPTIONAL)
```bash
python database/seed_admin.py
```

### BƯỚC 5: CHẠY APP
```bash
python run.py
```

### BƯỚC 6: TEST REGISTER
Truy cập: `http://localhost:5000/auth/register`

## 🔧 Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

**Hoặc cài thủ công:**
```bash
pip install flask-wtf wtforms email-validator
```

## 🐛 Lỗi Thường Gặp

### ❌ "Không tìm thấy role học viên"
**Nguyên nhân:** Chưa chạy seed roles
**Giải pháp:** 
```bash
python database/seed_roles.py
```

### ❌ "CSRF token missing"  
**Nguyên nhân:** CSRF protection
**Giải pháp:** Trong `app/__init__.py` thêm:
```python
app.config['WTF_CSRF_ENABLED'] = False
```

### ❌ "No module named 'app'"
**Nguyên nhân:** Chạy từ sai thư mục
**Giải pháp:** Chạy từ thư mục root project

### ❌ "Can't connect to MySQL"
**Nguyên nhân:** MySQL chưa chạy hoặc connection string sai
**Giải pháp:** 
1. Kiểm tra MySQL đang chạy
2. Sửa connection string trong `app/config.py`

## 🔍 Kiểm tra Roles Đã Tồn Tại

```bash
mysql -u root -p
USE AI_WRTS;
SELECT * FROM roles;
```

**Phải thấy 4 rows:**
```
+------+---------------+-----------+
| id   | role_name     | role_code |
+------+---------------+-----------+
| 1    | Student       | STUDENT   |
| 2    | Instructor    | INSTRUCTOR|
| 3    | Administrator | ADMIN     |
| 4    | Manager       | MANAGER   |
+------+---------------+-----------+
```

**Nếu chưa có roles → Chạy ngay:**
```bash
python database/seed_roles.py
```

## 📁 Cấu trúc Project

```
ai-wrts-system/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── config.py            # Configuration
│   ├── models/              # Database models
│   ├── routes/              # Route handlers
│   ├── services/            # Business logic
│   ├── forms/               # Flask-WTF forms
│   └── utils/               # Utilities & decorators
├── templates/               # Jinja2 templates
├── database/                # Database scripts
├── run.py                   # Application entry point
└── requirements.txt         # Dependencies
```

## 🎯 Features

- ✅ **Authentication System** - Login/Register với Flask-WTF
- ✅ **Role-based Access** - Student, Instructor, Admin, Manager
- ✅ **Form Validation** - Client & Server-side validation
- ✅ **CSRF Protection** - Bảo mật form submission
- ✅ **Database Migration** - Flask-Migrate integration
- ✅ **Session Management** - Secure user sessions

## 🔗 Links Truy cập

- **Login:** `http://localhost:5000/auth/login`
- **Register:** `http://localhost:5000/auth/register`
- **Admin Dashboard:** `http://localhost:5000/admin/dashboard`
- **Student Dashboard:** `http://localhost:5000/student/dashboard`
- **Instructor Dashboard:** `http://localhost:5000/instructor/dashboard`
- **Manager Dashboard:** `http://localhost:5000/manager/dashboard`

## 🛠️ Development

### Debug Mode
```bash
# Tắt CSRF để debug
# Trong app/__init__.py uncomment:
app.config['WTF_CSRF_ENABLED'] = False
```

### Database Reset
```bash
# Xóa migrations và tạo lại
rm -rf migrations/
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
python database/seed_roles.py
```

## 📝 Notes

- **SECRET_KEY:** Được set trong `app/config.py`
- **Database:** MySQL với PyMySQL driver
- **Templates:** Jinja2 với Flask-WTF form integration
- **Security:** CSRF protection, password hashing với Werkzeug
