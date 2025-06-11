import matplotlib
matplotlib.use('Agg')  # Использовать backend без GUI
import matplotlib.pyplot as plt
import numpy as np

# Входные данные
input_points = [
    {
        "x": 0,
        "y": 0,
        "t": 0
    },
    {
        "x": 2,
        "y": 3,
        "t": 3
    },
    {
        "x": 4,
        "y": 4,
        "t": 4
    },
    {
        "x": 6,
        "y": 5,
        "t": 6
    },
    {
        "x": 7,
        "y": 8,
        "t": 9
    }
]

# Предсказанная точка
predicted_point = {
    "x": 7.0118865966796875,
    "y": 3.2640841007232666,
    "t": 4.628136157989502
}

# Преобразуем в массивы numpy для удобства
input_array = np.array([[p["x"], p["y"], p["t"]] for p in input_points])
pred_array = np.array([predicted_point["x"], predicted_point["y"], predicted_point["t"]])

def print_analysis():
    """Текстовый анализ предсказания"""
    print("🚁 === АНАЛИЗ ПРЕДСКАЗАНИЯ ТРАЕКТОРИИ БПЛА ===", flush=True)
    print(flush=True)
    
    print("📊 Входные точки:")
    for i, point in enumerate(input_points):
        print(f"  Точка {i+1}: X={point['x']:>3}, Y={point['y']:>3}, T={point['t']:>3}")
    
    print(f"\n🎯 Предсказанная точка:")
    print(f"  X = {pred_array[0]:>8.3f} м")
    print(f"  Y = {pred_array[1]:>8.3f} м") 
    print(f"  T = {pred_array[2]:>8.3f} с")
    
    # Анализ изменений
    print(f"\n📈 Анализ изменений:")
    last_input = input_points[-1]
    delta_x = pred_array[0] - last_input["x"]
    delta_y = pred_array[1] - last_input["y"]
    delta_t = pred_array[2] - last_input["t"]
    
    print(f"  ΔX = {delta_x:>8.3f} м (от последней точки)")
    print(f"  ΔY = {delta_y:>8.3f} м")
    print(f"  ΔT = {delta_t:>8.3f} с")

def plot_trajectory():
    """Построение графика траектории"""
    fig = plt.figure(figsize=(15, 10))
    
    # 2D график XY
    ax1 = fig.add_subplot(221)
    
    # Входные точки
    ax1.scatter(input_array[:, 0], input_array[:, 1], 
               c='blue', s=100, alpha=0.8, label='Входные точки', marker='o')
    
    # Предсказанная точка
    ax1.scatter(pred_array[0], pred_array[1], 
               c='red', s=150, alpha=0.9, label='Предсказание', marker='^')
    
    # Соединяем линией
    ax1.plot(input_array[:, 0], input_array[:, 1], 
            'b--', alpha=0.6, linewidth=2)
    
    # Линия к предсказанной точке
    last_point = input_array[-1]
    ax1.plot([last_point[0], pred_array[0]], 
            [last_point[1], pred_array[1]], 
            'r-', linewidth=3, alpha=0.8)
    
    # Добавляем номера точек
    for i, point in enumerate(input_points):
        ax1.annotate(f'{i+1}', (point["x"], point["y"]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    ax1.annotate('PRED', (pred_array[0], pred_array[1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10, color='red')
    
    ax1.set_xlabel('X (м)')
    ax1.set_ylabel('Y (м)')
    ax1.set_title('Траектория XY')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # График координаты X во времени
    ax2 = fig.add_subplot(222)
    t_values = [p["t"] for p in input_points]
    x_values = [p["x"] for p in input_points]
    
    ax2.plot(t_values, x_values, 'bo-', linewidth=2, markersize=8, label='X входные')
    ax2.plot(pred_array[2], pred_array[0], 'r^', markersize=12, label='X предсказание')
    ax2.plot([t_values[-1], pred_array[2]], [x_values[-1], pred_array[0]], 'r--', alpha=0.7)
    
    ax2.set_xlabel('Время (с)')
    ax2.set_ylabel('X (м)')
    ax2.set_title('Координата X во времени')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # График координаты Y во времени
    ax3 = fig.add_subplot(223)
    y_values = [p["y"] for p in input_points]
    
    ax3.plot(t_values, y_values, 'go-', linewidth=2, markersize=8, label='Y входные')
    ax3.plot(pred_array[2], pred_array[1], 'r^', markersize=12, label='Y предсказание')
    ax3.plot([t_values[-1], pred_array[2]], [y_values[-1], pred_array[1]], 'r--', alpha=0.7)
    
    ax3.set_xlabel('Время (с)')
    ax3.set_ylabel('Y (м)')
    ax3.set_title('Координата Y во времени')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # График времени (временная последовательность)
    ax4 = fig.add_subplot(224)
    steps = list(range(len(input_points)))
    
    ax4.plot(steps, t_values, 'co-', linewidth=2, markersize=8, label='T входные')
    ax4.plot(len(steps), pred_array[2], 'r^', markersize=12, label='T предсказание')
    ax4.plot([steps[-1], len(steps)], [t_values[-1], pred_array[2]], 'r--', alpha=0.7)
    
    ax4.set_xlabel('Шаг')
    ax4.set_ylabel('Время (с)')
    ax4.set_title('Временная последовательность')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('trajectory_prediction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ График сохранен как 'trajectory_prediction.png'")

def main():
    """Основная функция"""
    print_analysis()
    print("\n📊 Строим график...")
    plot_trajectory()
    print("\n🎉 Визуализация завершена!")

if __name__ == "__main__":
    main()
