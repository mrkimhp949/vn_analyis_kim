
import sqlite3
import queue
import threading
import logging
from typing import Any, List, Tuple, Callable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(message)s')

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path="trading_bot.db"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: str = "trading_bot.db"):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            
            self.db_path = db_path
            self.write_queue = queue.Queue()
            self.stop_event = threading.Event()
            
            # Use thread-local storage for read connections
            self.thread_local = threading.local()

            # Start the writer thread
            self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True, name="DBWriterThread")
            self.writer_thread.start()
            
            self._initialized = True
            logging.info(f"DatabaseManager initialized with db_path: {self.db_path}")

    def _get_read_conn(self):
        """Gets a read-only connection for the current thread."""
        if not hasattr(self.thread_local, "connection"):
            try:
                # URI for read-only mode to prevent accidental writes
                db_uri = f"file:{self.db_path}?mode=ro"
                self.thread_local.connection = sqlite3.connect(db_uri, uri=True, check_same_thread=False)
            except sqlite3.OperationalError:
                # Fallback for older SQLite versions that don't support URI
                self.thread_local.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        return self.thread_local.connection

    def _writer_worker(self):
        """The worker function that processes the write queue."""
        # The writer thread has its own dedicated connection
        try:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            logging.info("Writer thread started and connection established.")
        except Exception as e:
            logging.error(f"Writer thread failed to connect to database: {e}")
            return

        while not self.stop_event.is_set():
            try:
                # Wait for a task. Timeout allows the thread to check the stop_event periodically.
                query, params, callback = self.write_queue.get(timeout=1)
                
                try:
                    cursor = conn.cursor()
                    cursor.execute(query, params)
                    conn.commit()
                    if callback:
                        callback(cursor.lastrowid, None)
                except Exception as e:
                    logging.error(f"Error executing write query: {query} with params: {params} - {e}")
                    conn.rollback()
                    if callback:
                        callback(None, e)
                finally:
                    self.write_queue.task_done()

            except queue.Empty:
                # This is expected when the queue is empty, just continue
                continue
        
        conn.close()
        logging.info("Writer thread stopped and connection closed.")

    def execute_read(self, query: str, params: Tuple = ()) -> List[Tuple]:
        """Executes a read (SELECT) query."""
        try:
            conn = self._get_read_conn()
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except Exception as e:
            logging.error(f"Error executing read query: {query} - {e}")
            return []

    def execute_write(self, query: str, params: Tuple = ()):
        """
        Adds a write (INSERT, UPDATE, DELETE) query to the queue.
        This method is non-blocking.
        """
        self.write_queue.put((query, params, None))

    def execute_write_with_callback(self, query: str, params: Tuple = (), callback: Callable[[Any, Exception | None], None] = None):
        """
        Adds a write query to the queue and calls the callback upon completion.
        The callback receives (last_row_id, error).
        """
        self.write_queue.put((query, params, callback))

    def close(self):
        """Stops the writer thread and closes connections."""
        logging.info("Closing DatabaseManager...")
        self.stop_event.set()
        self.writer_thread.join(timeout=5) # Wait for the writer to finish
        if hasattr(self.thread_local, "connection"):
            self.thread_local.connection.close()
            logging.info("Read connection for the main thread closed.")
        logging.info("DatabaseManager closed.")

# Singleton instance
db_manager = DatabaseManager()
