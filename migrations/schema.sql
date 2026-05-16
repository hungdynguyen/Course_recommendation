-- ============================================================
-- VietCV Course Recommendation System – MySQL Schema
-- ============================================================
-- Run: mysql -u root -p vietcv < migrations/schema.sql
-- Or auto-run via docker-compose init script
-- ============================================================

SET NAMES utf8mb4;
SET CHARACTER SET utf8mb4;

-- -----------------------------------------------------------
-- 1. Course versions (must be created BEFORE courses)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS course_versions (
    version_id   VARCHAR(50) PRIMARY KEY,
    course_count INT DEFAULT 0,
    status       VARCHAR(20) NOT NULL DEFAULT 'building'
                 COMMENT 'building | ready | archived',
    es_index     VARCHAR(255) COMMENT 'Elasticsearch index name for this version',
    s3_backup_path VARCHAR(500) COMMENT 'S3 backup path',
    checksum     VARCHAR(64),
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    archived_at  TIMESTAMP NULL,
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------
-- 2. Courses
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS courses (
    id           VARCHAR(50) PRIMARY KEY,
    title        VARCHAR(255) NOT NULL,
    description  LONGTEXT,
    skills       JSON COMMENT 'Array of skill names extracted from course',
    content_url  VARCHAR(500) COMMENT 'S3 path to original docx/pdf',
    version_id   VARCHAR(50),
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at   TIMESTAMP NULL,
    INDEX idx_version_id (version_id),
    INDEX idx_deleted_at (deleted_at),
    FOREIGN KEY (version_id) REFERENCES course_versions(version_id)
        ON DELETE SET NULL ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------
-- 3. Upload batches
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS upload_batches (
    batch_id         VARCHAR(50) PRIMARY KEY,
    file_count       INT DEFAULT 0,
    total_size_bytes BIGINT DEFAULT 0,
    status           VARCHAR(20) NOT NULL DEFAULT 'pending'
                     COMMENT 'pending | processing | completed | failed',
    error_message    LONGTEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at     TIMESTAMP NULL,
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------
-- 4. Pipeline runs (Airflow DAG runs)
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id           VARCHAR(50) PRIMARY KEY,
    batch_id         VARCHAR(50),
    status           VARCHAR(20) NOT NULL DEFAULT 'running'
                     COMMENT 'running | completed | failed',
    version_id       VARCHAR(50),
    progress_percent INT DEFAULT 0,
    error_log        LONGTEXT,
    started_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at     TIMESTAMP NULL,
    FOREIGN KEY (batch_id) REFERENCES upload_batches(batch_id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    FOREIGN KEY (version_id) REFERENCES course_versions(version_id)
        ON DELETE SET NULL ON UPDATE CASCADE,
    INDEX idx_status (status),
    INDEX idx_version_id (version_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------
-- 5. CV profiles
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS cv_profiles (
    cv_id            VARCHAR(50) PRIMARY KEY,
    filename         VARCHAR(255),
    parsed_json      JSON,
    skills           JSON,
    experience_years INT,
    s3_path          VARCHAR(500),
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------
-- 6. Job descriptions
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS job_descriptions (
    jd_id            VARCHAR(50) PRIMARY KEY,
    title            VARCHAR(255),
    parsed_json      JSON,
    required_skills  JSON,
    s3_path          VARCHAR(500),
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- -----------------------------------------------------------
-- 7. Audit logs
-- -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_logs (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    action        VARCHAR(50) NOT NULL
                  COMMENT 'create | update | delete | upload | process',
    resource_type VARCHAR(50) NOT NULL
                  COMMENT 'course | batch | version | pipeline',
    resource_id   VARCHAR(50),
    user_id       VARCHAR(50),
    details       JSON,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_resource (resource_type, resource_id),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
