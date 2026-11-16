from .database import get_connection


def remember(key: str, value: str):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO memories(key, value)
        VALUES(?, ?)
    """, (key, value))

    conn.commit()
    conn.close()


def recall(key: str):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT value
        FROM memories
        WHERE key=?
    """, (key,))

    row = cursor.fetchone()

    conn.close()

    if row:
        return row[0]

    return None