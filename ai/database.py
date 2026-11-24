import sqlite3
from datetime import datetime

DB_NAME = "crowd_data.db"


def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crowd_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            people_in_frame INTEGER,
            entry_count INTEGER,
            exit_count INTEGER,
            occupancy INTEGER,
            density TEXT,
            alert_status TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_crowd_data(
    people_in_frame,
    entry_count,
    exit_count,
    occupancy,
    density,
    alert_status
):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO crowd_data (
            timestamp,
            people_in_frame,
            entry_count,
            exit_count,
            occupancy,
            density,
            alert_status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        people_in_frame,
        entry_count,
        exit_count,
        occupancy,
        density,
        alert_status
    ))

    conn.commit()
    conn.close()


def get_recent_data(limit=20):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT *
        FROM crowd_data
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    data = cursor.fetchall()

    conn.close()

    return data


if __name__ == "__main__":
    create_database()
    print("Database created successfully!")