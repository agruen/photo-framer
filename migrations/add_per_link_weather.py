#!/usr/bin/env python3
"""
Migration: Add per-link weather configuration to SlideshowURL

This migration adds weather_zip and weather_api_key columns to the
slideshow_url table to support per-link weather overrides.
"""

import sqlite3
import sys
import os

# Add parent directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.config import Config


def migrate():
    """Apply the migration"""
    db_path = Config.SQLALCHEMY_DATABASE_URI.replace('sqlite:///', '')
    print(f"Migrating database at: {db_path}")

    # Check if database exists
    if not os.path.exists(db_path):
        print("ERROR: Database does not exist. Run the application first to create it.")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(slideshow_url)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'weather_zip' in columns and 'weather_api_key' in columns:
            print("Migration already applied - columns already exist.")
            return True

        # Add weather_zip column
        if 'weather_zip' not in columns:
            print("Adding weather_zip column...")
            cursor.execute("ALTER TABLE slideshow_url ADD COLUMN weather_zip VARCHAR(20)")
            print("✓ Added weather_zip column")

        # Add weather_api_key column
        if 'weather_api_key' not in columns:
            print("Adding weather_api_key column...")
            cursor.execute("ALTER TABLE slideshow_url ADD COLUMN weather_api_key VARCHAR(100)")
            print("✓ Added weather_api_key column")

        conn.commit()
        print("\nMigration completed successfully!")
        return True

    except Exception as e:
        print(f"ERROR during migration: {e}")
        conn.rollback()
        return False

    finally:
        conn.close()


def rollback():
    """Rollback the migration (SQLite doesn't support DROP COLUMN easily)"""
    print("WARNING: SQLite does not support DROP COLUMN.")
    print("To rollback, you would need to:")
    print("1. Create a new table without the weather columns")
    print("2. Copy data from old table to new table")
    print("3. Drop old table and rename new table")
    print("\nThis is not implemented automatically for safety.")
    return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Database migration for per-link weather')
    parser.add_argument('action', choices=['migrate', 'rollback'],
                       help='Action to perform')
    args = parser.parse_args()

    if args.action == 'migrate':
        success = migrate()
        sys.exit(0 if success else 1)
    elif args.action == 'rollback':
        rollback()
        sys.exit(1)
