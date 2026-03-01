#!/usr/bin/env python3
"""
Script thống kê các khóa học trong Data_Courses_Json
"""

import json
from pathlib import Path
import numpy as np
from collections import defaultdict

# Cấu hình
DATA_DIR = Path("./data/Data_Courses_Json")

print("=" * 80)
print("📚 THỐNG KÊ KHÓA HỌC")
print("=" * 80)

# Thu thập dữ liệu
courses = []
all_skills_taught = []
all_skills_required = []
courses_with_requirements = []

print(f"\n📂 Đang đọc khóa học từ {DATA_DIR}...\n")

# Duyệt qua tất cả các khoa
for faculty_dir in sorted(DATA_DIR.iterdir()):
    if not faculty_dir.is_dir():
        continue
    
    faculty_name = faculty_dir.name
    print(f"  📁 {faculty_name}")
    
    course_count = 0
    # Duyệt qua các file JSON trong khoa
    for course_file in sorted(faculty_dir.glob("*.json")):
        try:
            with open(course_file, 'r', encoding='utf-8') as f:
                course_data = json.load(f)
            
            # Lấy thông tin skills (cấu trúc thực tế)
            skills_taught = course_data.get('skill_outcomes', [])
            entry_requirements = course_data.get('entry_requirements', {})
            skills_required = entry_requirements.get('minimum_entry_skills', [])
            
            # Lưu thông tin
            courses.append({
                'name': course_data.get('title', course_file.stem),
                'code': course_data.get('course_id', ''),
                'faculty': faculty_name,
                'skills_taught': len(skills_taught),
                'skills_required': len(skills_required)
            })
            
            all_skills_taught.append(len(skills_taught))
            all_skills_required.append(len(skills_required))
            
            if len(skills_required) > 0:
                courses_with_requirements.append(len(skills_required))
            
            course_count += 1
            
        except Exception as e:
            print(f"    ⚠️  Lỗi đọc {course_file.name}: {e}")
    
    print(f"      → {course_count} khóa học")

print(f"\n✓ Tổng cộng đọc được {len(courses)} khóa học từ {len(list(DATA_DIR.iterdir()))} khoa\n")

# ============== THỐNG KÊ ==============
print("=" * 80)
print("📊 THỐNG KÊ CHI TIẾT")
print("=" * 80)

print(f"\n🎓 TỔNG QUAN:")
print(f"  • Tổng số khóa học: {len(courses)}")
print(f"  • Số khoa: {len(set(c['faculty'] for c in courses))}")

# Skills được dạy (skills_taught)
print(f"\n📚 SKILLS ĐƯỢC DẠY (per course):")
print(f"  • Trung bình: {np.mean(all_skills_taught):.2f} skills/khóa học")
print(f"  • Median: {np.median(all_skills_taught):.0f} skills")
print(f"  • Tối thiểu: {np.min(all_skills_taught)} skills")
print(f"  • Tối đa: {np.max(all_skills_taught)} skills")
print(f"  • Std: {np.std(all_skills_taught):.2f}")
print(f"  • Q25: {np.percentile(all_skills_taught, 25):.0f} skills")
print(f"  • Q75: {np.percentile(all_skills_taught, 75):.0f} skills")

# Tìm khóa học dạy nhiều skills nhất
top_taught = sorted(courses, key=lambda x: x['skills_taught'], reverse=True)[:5]
print(f"\n  📌 Top 5 khóa học dạy nhiều skills nhất:")
for i, course in enumerate(top_taught, 1):
    print(f"     {i}. {course['name'][:50]:50s} - {course['skills_taught']} skills")

# Skills yêu cầu (required_skills)
print(f"\n📋 SKILLS YÊU CẦU (per course):")
print(f"  • Trung bình (tất cả khóa học): {np.mean(all_skills_required):.2f} skills/khóa học")
print(f"  • Median (tất cả): {np.median(all_skills_required):.0f} skills")
print(f"  • Tối thiểu: {np.min(all_skills_required)} skills")
print(f"  • Tối đa: {np.max(all_skills_required)} skills")

courses_no_requirements = len([s for s in all_skills_required if s == 0])
courses_has_requirements = len(courses) - courses_no_requirements

print(f"\n  📊 Phân loại:")
print(f"     • Khóa học KHÔNG có yêu cầu: {courses_no_requirements} ({courses_no_requirements/len(courses)*100:.1f}%)")
print(f"     • Khóa học CÓ yêu cầu: {courses_has_requirements} ({courses_has_requirements/len(courses)*100:.1f}%)")

if len(courses_with_requirements) > 0:
    print(f"\n  🎯 Chỉ tính các khóa học CÓ điều kiện đầu vào:")
    print(f"     • Trung bình: {np.mean(courses_with_requirements):.2f} skills/khóa học")
    print(f"     • Median: {np.median(courses_with_requirements):.0f} skills")
    print(f"     • Tối thiểu: {np.min(courses_with_requirements)} skills")
    print(f"     • Tối đa: {np.max(courses_with_requirements)} skills")
    print(f"     • Std: {np.std(courses_with_requirements):.2f}")

# Tìm khóa học yêu cầu nhiều skills nhất
top_required = sorted(courses, key=lambda x: x['skills_required'], reverse=True)[:5]
print(f"\n  📌 Top 5 khóa học yêu cầu nhiều skills nhất:")
for i, course in enumerate(top_required, 1):
    if course['skills_required'] > 0:
        print(f"     {i}. {course['name'][:50]:50s} - {course['skills_required']} skills")

# Phân phối theo khoa
print(f"\n🏫 PHÂN PHỐI THEO KHOA:")
faculty_stats = defaultdict(lambda: {'count': 0, 'taught': [], 'required': []})

for course in courses:
    faculty = course['faculty']
    faculty_stats[faculty]['count'] += 1
    faculty_stats[faculty]['taught'].append(course['skills_taught'])
    faculty_stats[faculty]['required'].append(course['skills_required'])

print(f"\n{'Khoa':<50s} {'Khóa học':>10s} {'TB Skills dạy':>15s} {'TB Skills cần':>15s}")
print("-" * 92)

for faculty in sorted(faculty_stats.keys()):
    stats = faculty_stats[faculty]
    faculty_display = faculty.replace('_2024', '')[:48]
    print(f"{faculty_display:<50s} {stats['count']:>10d} {np.mean(stats['taught']):>15.2f} {np.mean(stats['required']):>15.2f}")

# Tư vấn và insights
print(f"\n" + "=" * 80)
print("💡 INSIGHTS & NHẬN XÉT")
print("=" * 80)

avg_taught = np.mean(all_skills_taught)
avg_required_all = np.mean(all_skills_required)
avg_required_filtered = np.mean(courses_with_requirements) if courses_with_requirements else 0

print(f"\n1. Về Skills được dạy:")
if avg_taught < 5:
    print(f"   ⚠️  Trung bình {avg_taught:.1f} skills/khóa → Khá ít")
elif avg_taught < 10:
    print(f"   ✓ Trung bình {avg_taught:.1f} skills/khóa → Hợp lý")
else:
    print(f"   ✅ Trung bình {avg_taught:.1f} skills/khóa → Phong phú")

print(f"\n2. Về Skills yêu cầu:")
print(f"   • {courses_no_requirements/len(courses)*100:.1f}% khóa học không có yêu cầu đầu vào")
print(f"   • Các khóa học CÓ yêu cầu thì trung bình cần {avg_required_filtered:.1f} skills")

if courses_no_requirements / len(courses) > 0.5:
    print(f"   → Đa số là khóa học cơ bản/nhập môn")
else:
    print(f"   → Hệ thống có cấu trúc tiên quyết rõ ràng")

print(f"\n3. Tỷ lệ taught/required:")
if avg_required_filtered > 0:
    ratio = avg_taught / avg_required_filtered
    print(f"   • Ratio = {ratio:.2f}")
    if ratio > 2:
        print(f"   → Mỗi khóa học dạy gấp {ratio:.1f}x số skills cần có")
        print(f"   → Sinh viên tích lũy kiến thức nhanh")

# Lưu kết quả
output_file = Path("course_statistics.json")
output_data = {
    'summary': {
        'total_courses': len(courses),
        'total_faculties': len(set(c['faculty'] for c in courses)),
        'skills_taught': {
            'mean': float(np.mean(all_skills_taught)),
            'median': float(np.median(all_skills_taught)),
            'min': int(np.min(all_skills_taught)),
            'max': int(np.max(all_skills_taught)),
            'std': float(np.std(all_skills_taught))
        },
        'skills_required_all': {
            'mean': float(np.mean(all_skills_required)),
            'median': float(np.median(all_skills_required)),
            'min': int(np.min(all_skills_required)),
            'max': int(np.max(all_skills_required))
        },
        'skills_required_filtered': {
            'mean': float(avg_required_filtered),
            'median': float(np.median(courses_with_requirements)) if courses_with_requirements else 0,
            'min': int(np.min(courses_with_requirements)) if courses_with_requirements else 0,
            'max': int(np.max(courses_with_requirements)) if courses_with_requirements else 0,
            'count': len(courses_with_requirements)
        },
        'courses_with_no_requirements': courses_no_requirements,
        'courses_with_requirements': courses_has_requirements
    },
    'details': courses
}

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print(f"\n📁 Đã lưu kết quả chi tiết vào: {output_file}")

print("\n" + "=" * 80)
print("✅ HOÀN TẤT!")
print("=" * 80)
