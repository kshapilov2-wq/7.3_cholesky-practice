#include <iostream>     
#include <vector>       
#include <string>       
#include <random>       
#include <iomanip>      
#include <cmath>        
using namespace std;                  

// ---------- Типы и счётчик операций ----------
using Matrix = vector<vector<double>>; 
using Vector = vector<double>;         

// Структура для подсчёта элементарных операций 
struct Ops {
	long long add = 0;               // Сложения/вычитания
	long long mul = 0;               // Умножения
	long long divv = 0;               // Деления
	long long sqrtv = 0;               // Извлечения квадратного корня
	void reset() { add = mul = divv = sqrtv = 0; } // Сброс всех счётчиков к нулю
};

// ---------- Утилиты вывода ----------
static void print_matrix(const Matrix& A, const string& name) { // Печать матрицы с названием
	cout << name << " =\n";             // Заголовок
	cout.setf(ios::fixed);              // Фиксированный формат чисел с плавающей точкой
	cout << setprecision(6);            // Шесть знаков после запятой
	for (const auto& row : A) {          // Идём по строкам матрицы
		for (double v : row)            // Идём по элементам строки
			cout << setw(12) << v << ' '; // Печатаем выровненно по ширине 12
		cout << '\n';                   // Перевод строки после каждой строки матрицы
	}
}
static void print_vector(const Vector& v, const string& name) { // Печать вектора с названием
	cout.setf(ios::fixed);              // Фиксированный формат вывода
	cout << setprecision(6);            // Шесть знаков после запятой
	cout << name << " = [ ";            // Заголовок и открывающая скобка
	for (double x : v)                  // Идём по элементам вектора
		cout << x << ' ';               // Печатаем значение и пробел
	cout << "]^T\n";                    // Закрывающая скобка и обозначение вектора-столбца
}

// ---------- Базовые операции линейной алгебры ----------
static bool is_symmetric(const Matrix& A, double tol = 1e-12) { // Проверка симметрии A=A^T с допуском tol
	size_t n = A.size();               // Размерность матрицы
	for (size_t i = 0; i < n; ++i)           // Внешний цикл по строкам
		for (size_t j = i + 1; j < n; ++j)     // Внутренний цикл по столбцам над диагональю
			if (fabs(A[i][j] - A[j][i]) > tol) // Если элемент и транспонированный заметно различаются
				return false;          // То матрица несимметрична
	return true;                       // Иначе — симметрична
}

static Vector matvec(const Matrix& A, const Vector& x, Ops* ops = nullptr) { // y = A x (с опциональным подсчётом операций)
	size_t n = A.size();               // Размерность
	Vector y(n, 0.0);                  // Результат инициализируем нулями
	for (size_t i = 0; i < n; ++i) {          // Идём по строкам A
		double s = 0.0;                // Аккумулятор скалярного произведения
		for (size_t j = 0; j < n; ++j) {      // Идём по столбцам A
			s += A[i][j] * x[j];       // Накопление суммы A[i][j]*x[j]
			if (ops) { ops->mul++; ops->add++; } // Учитываем одно умножение и одно сложение
		}
		y[i] = s;                      // Записываем элемент результата
	}
	return y;                          // Возвращаем вектор y
}

static double norm2(const Vector& v) {  // Евклидова норма вектора
	double s = 0.0;                    // Сумма квадратов
	for (double x : v)                 // Идём по элементам
		s += x * x;                      // Добавляем квадрат
	return sqrt(s);                    // Возвращаем корень из суммы квадратов
}

static double residual_norm2(const Matrix& A, const Vector& x, const Vector& b) { // ||Ax-b||_2
	Vector r = matvec(A, x, nullptr);  // Считаем r = A x (без подсчёта операций)
	for (size_t i = 0; i < r.size(); ++i)    // Идём по компонентам
		r[i] -= b[i];                  // Вычитаем b
	return norm2(r);                   // Возвращаем норму r
}

// ---------- Метод Холецкого: A = L L^T ----------
static bool cholesky_decompose(const Matrix& A, Matrix& L, Ops& ops, double tol = 1e-14) { // Разложение A на L L^T
	size_t n = A.size();               // Размерность
	L.assign(n, Vector(n, 0.0));       // Обнуляем матрицу L (нижняя треугольная)

	for (size_t k = 0; k < n; ++k) {        // Главный цикл по столбцам/строкам k
		double sumsq = 0.0;            // Сумма квадратов L[k][s]^2 для s=0..k-1
		for (size_t s = 0; s < k; ++s) {    // Бежим по уже посчитанным элементам строки k
			sumsq += L[k][s] * L[k][s];  // Добавляем квадрат элемента
			ops.mul++; ops.add++;      // Учитываем умножение и сложение
		}
		double diag = A[k][k] - sumsq; // Подкоренное значение на диагонали
		ops.add++;                     // Учитываем вычитание как «add»
		if (diag <= tol)               // Если значение не положительно  —
			return false;              // разложение Холецкого невозможно 

		L[k][k] = sqrt(diag);          // Диагональный элемент L[k][k] = sqrt(diag)
		ops.sqrtv++;                   // Учитываем операцию извлечения корня

		for (size_t i = k + 1; i < n; ++i) {  // Заполняем элементы ниже диагонали в столбце k
			double s = 0.0;            // Сумма L[i][t]*L[k][t] для t=0..k-1
			for (size_t t = 0; t < k; ++t) { // Бежим по уже посчитанным столбцам
				s += L[i][t] * L[k][t];  // Накопление произведений
				ops.mul++; ops.add++;  // Учёт операций
			}
			double num = A[i][k] - s;  // Числитель формулы для l_{ik}
			ops.add++;                 // Учёт вычитания
			L[i][k] = num / L[k][k];   // Делим на диагональный элемент l_{kk}
			ops.divv++;                // Учёт деления
		}
	}
	return true;                       
}

static Vector forward_subst(const Matrix& L, const Vector& b, Ops& ops) { // Решаем L y = b (нижняя треугольная)
	size_t n = L.size();               // Размерность
	Vector y(n, 0.0);                  // Результат
	for (size_t i = 0; i < n; ++i) {          // Идём сверху вниз
		double s = 0.0;                // Сумма L[i][j]*y[j] по j<i
		for (size_t j = 0; j < i; ++j) {      // Только уже известные компоненты
			s += L[i][j] * y[j];         // Накопление суммы
			ops.mul++; ops.add++;      // Учёт операций
		}
		double num = b[i] - s;         // Числитель
		ops.add++;                     // Учёт вычитания
		y[i] = num / L[i][i];          // Деление на диагональный элемент
		ops.divv++;                    // Учёт деления
	}
	return y;                          // Возвращаем вектор y
}

static Vector back_subst_LT(const Matrix& L, const Vector& y, Ops& ops) { // Решаем L^T x = y 
	size_t n = L.size();               // Размерность
	Vector x(n, 0.0);                  // Результат
	for (int ii = (int)n - 1; ii >= 0; --ii) { // Идём снизу вверх
		size_t i = (size_t)ii;         // Текущий индекс как size_t
		double s = 0.0;                // Сумма L[j][i]*x[j] по j>i
		for (size_t j = i + 1; j < n; ++j) {    // Колонки справа (в L^T это над диагональю)
			s += L[j][i] * x[j];         // Накопление суммы
			ops.mul++; ops.add++;      // Учёт операций
		}
		double num = y[i] - s;         // Числитель
		ops.add++;                     // Учёт вычитания
		x[i] = num / L[i][i];          // Делим на диагональный элемент (тот же, что в L)
		ops.divv++;                    // Учёт деления
	}
	return x;                          // Возвращаем вектор x
}

static bool solve_cholesky(const Matrix& A, const Vector& b, // Комплексная «обёртка» решения Ax=b через Холецкого
	Vector& x, Matrix& L, Ops& ops) {
	ops.reset();                       // Сбрасываем счётчики операций
	if (!is_symmetric(A))              // Если A не симметрична,
		return false;                  
	if (!cholesky_decompose(A, L, ops))// Пытаемся построить A=L L^T 
		return false;                  
	Vector y = forward_subst(L, b, ops); // Решаем L y = b
	x = back_subst_LT(L, y, ops);      // Решаем L^T x = y
	return true;                       // Успех
}

// ---------- Генерация тестовых матриц ----------
static Matrix make_SPD(size_t n, unsigned seed = 42, double tau = 1.0) { // Строим A = M^T M + tau*I
	mt19937_64 rng(seed);                 // Инициализируем Генератор случайных чисел
	uniform_real_distribution<double> d(-1.0, 1.0); // Равномерное распределение
	Matrix M(n, Vector(n, 0.0));          // Пустая матрица M
	for (size_t i = 0; i < n; ++i)              // Заполняем M случайными значениями
		for (size_t j = 0; j < n; ++j)
			M[i][j] = d(rng);
	Matrix A(n, Vector(n, 0.0));          // Будущая симетричная, положительно-определенная-матрица
	for (size_t i = 0; i < n; ++i)              // Считаем A = M^T M
		for (size_t j = 0; j < n; ++j) {
			double s = 0.0;                 // Скалярное произведение столбцов i и j
			for (size_t k = 0; k < n; ++k)      // Суммируем по k
				s += M[k][i] * M[k][j];     // Добавляем M[k][i]*M[k][j]
			A[i][j] = s;                  // Записываем A[i][j]
		}
	for (size_t i = 0; i < n; ++i)              // Добавляем tau к диагонали
		A[i][i] += tau;                   // Это «укрепляет» положительную определённость
	return A;                             // Возвращаем матрицу
}

static Matrix make_sym_indef(size_t n, unsigned seed = 777) { // Строим симметричную, но обычно не положительно-определенную матрицу
	mt19937_64 rng(seed);                 // Инициализируем 
	uniform_real_distribution<double> d(-1.0, 1.0); // Равномерное распределение
	Matrix A(n, Vector(n, 0.0));          // Пустая матрица
	for (size_t i = 0; i < n; ++i)              // Заполняем верхний треугольник
		for (size_t j = i; j < n; ++j) {
			double v = d(rng);            // Случайное число
			A[i][j] = A[j][i] = v;        // Симметризуем
		}
	for (size_t i = 0; i < n; ++i)              // Сместим диагональ вниз
		A[i][i] -= 2.5;                   // Чтобы с высокой вероятностью нарушить положительную-определенность
	return A;
}

// ---------- Точка входа ----------
int main(int argc, char** argv) {        
	setlocale(LC_ALL, "Russian");
	ios::sync_with_stdio(false);          // Ускоряем ввод-вывод
	cin.tie(nullptr);                     // Отключаем синхронизацию с C I/O

	string variant = (argc >= 2 ? string(argv[1]) : string("spd")); // Читаем тип варианта: "spd" или "bad" (по умолчанию spd)
	size_t n = (argc >= 3 ? (size_t)stoul(argv[2]) : (size_t)5);    // Читаем размерность (по умолчанию 5)
	if (n < 5) n = 5;                       // Гарантируем требование n ≥ 5

	cout << "Вариант: " << variant << ", n=" << n << "\n"; // Сообщаем выбранные параметры

	Matrix A;                                // Матрица системы
	if (variant == "spd")                    // Если выбран SPD-вариант
		A = make_SPD(n, 42, 1.0);            // Генерируем SPD-матрицу
	else                                     // Иначе считаем, что нужен «плохой» вариант
		A = make_sym_indef(n, 777);          // Генерируем симметричную, но не-PD матрицу

	Vector xstar(n, 1.0);                    // Истинное решение (вектор из единиц) для контроля
	Vector b = matvec(A, xstar, nullptr);    // Строим правую часть b = A * x* (чтобы знать, к чему сравнивать)

	Matrix L;                                // Здесь будет нижняя треугольная L
	Vector x;                                // Здесь будет решение x
	Ops ops;                                 // Счётчик операций

	bool ok = solve_cholesky(A, b, x, L, ops); // Пытаемся решить через метод Холецкого

	if (ok) {                                  // Если метод применим и решение получено
		print_matrix(L, "L");                 // Печатаем матрицу L
		print_vector(x, "x");                 // Печатаем найденное решение x
		cout << "||Ax - b||_2 = "            // Печатаем норму невязки
			<< residual_norm2(A, x, b) << "\n";
		cout << "ops: add=" << ops.add        // Печатаем счётчики операций
			<< " mul=" << ops.mul
			<< " div=" << ops.divv
			<< " sqrt=" << ops.sqrtv
			<< " total~=" << (ops.add + ops.mul + ops.divv + ops.sqrtv) << "\n";
	}
	else {                                  // Если метод неприменим
		cout << "Разложение Холецкого НЕ применимо для данной матрицы (несимметрична или не-PD).\n";
		// В этом варианте решение не выводим
	}

	return 0;                                 // Завершаем программу
}
