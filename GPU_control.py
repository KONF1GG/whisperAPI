import fcntl
import time
import os
from contextlib import contextmanager

class GPULock:
    def __init__(self, lock_file_path="/shared/gpu.lock", timeout=None):
        """
        Инициализация менеджера блокировки GPU.
        :param lock_file_path: Путь к файлу блокировки
        :param timeout: Максимальное время ожидания в секундах (None = бесконечно)
        """
        self.lock_file_path = lock_file_path
        self.timeout = timeout
        self.lock_file = None

    def acquire(self):
        """Попытка установить блокировку"""
        start_time = time.time()
        self.lock_file = open(self.lock_file_path, "a")
        while True:
            try:
                fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                print(f"[{os.getpid()}] Блокировка GPU установлена")
                return True
            except IOError:
                if self.timeout is not None and (time.time() - start_time) > self.timeout:
                    print(f"[{os.getpid()}] Таймаут ожидания GPU истек")
                    return False
                print(f"[{os.getpid()}] GPU занят, жду...")
                time.sleep(1)

    def release(self):
        """Снятие блокировки"""
        if self.lock_file:
            fcntl.flock(self.lock_file, fcntl.LOCK_UN)
            self.lock_file.close()
            print(f"[{os.getpid()}] Блокировка GPU снята")

    def __enter__(self):
        """Вход в контекст"""
        if not self.acquire():
            raise RuntimeError("Не удалось захватить GPU")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекста"""
        self.release()

@contextmanager
def gpu_lock(timeout=None):
    lock = GPULock(timeout=timeout)
    with lock:
        yield