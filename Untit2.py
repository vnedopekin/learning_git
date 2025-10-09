import torch
import time

# Убедимся, что используем GPU
device = torch.device('cuda')

print("🎯 ЗАПУСК ТЕСТА НА NVIDIA")

# Создаем тензоры НАПРЯМУЮ на GPU
x = torch.randn(10000, 10000, device=device)  # Сразу на GPU!
y = torch.randn(10000, 10000, device=device)  # Сразу на GPU!

print(f"Тензор x на: {x.device}")
print(f"Тензор y на: {y.device}")

# Тестируем умножение матриц
start_time = time.time()
z = torch.matmul(x, y)
torch.cuda.synchronize()  # Ждем завершения на GPU
end_time = time.time()

print(f"✅ Умножение матриц 10000x10000 выполнено!")
print(f"⏱️ Время: {end_time - start_time:.2f} секунд")
print(f"📊 Результат на устройстве: {z.device}")

# Проверяем использование памяти
print(f"💾 Использовано памяти: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")