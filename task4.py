import math
import pylab
import numpy
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt

import boundary
from objects import LayerContinuous, LayerDiscrete, Probe

class Gaussian:
    '''
    Источник, создающий гауссов импульс
    '''

    def __init__(self, magnitude, dg, wg):

        self.magnitude = magnitude
        self.dg = dg
        self.wg = wg

    def getField(self, time):
        return self.magnitude * numpy.exp(-((time - self.dg) / self.wg) ** 2)
    
class Sampler:
    def __init__(self, discrete: float):
        self.discrete = discrete

    def sample(self, x: float) -> int:
        return math.floor(x / self.discrete + 0.5)


def sampleLayer(layer_cont, sampler):
    start_discrete = sampler.sample(layer_cont.xmin)
    end_discrete = (sampler.sample(layer_cont.xmax)
                    if layer_cont.xmax is not None
                    else None)
    return LayerDiscrete(start_discrete, end_discrete,
                         layer_cont.eps, layer_cont.mu)


def fillMedium(layer: LayerDiscrete, eps, mu):
    if layer.xmax is not None:
        eps[layer.xmin: layer.xmax] = layer.eps
        mu[layer.xmin: layer.xmax] = layer.mu
    else:
        eps[layer.xmin:] = layer.eps
        mu[layer.xmin:] = layer.mu


if __name__ == '__main__':
    # Используемые константы
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Скорость света в вакууме
    c = 299792458.0

    # Параметры моделирования
    # Дискрет по пространству в м
    dx = 2e-3
    
    # Число Куранта
    Sc = 1.0

    # Размер области моделирования в м
    maxSize_m = 0.5

    # Время расчета в секундах
    maxTime_s = 100e-9

    # Положение источника в м
    sourcePos_m = 0.1

    # Координаты датчикa для регистрации поля в м
    probePos_m = 0.05

    # Параметры слоев
    layers_cont = [LayerContinuous(xmin = 0.15, xmax = 0.25, eps=4.0),
                   LayerContinuous(xmin = 0.25, xmax = 0.3, eps=9.0),
                   LayerContinuous(xmin = 0.3, xmax = None, eps=1.0)]

    # Переход к дискретным отсчетам
    # Дискрет по времени
    dt = dx * Sc / c

    sampler_x = Sampler(dx)
    sampler_t = Sampler(dt)

    # Время расчета в отсчетах
    maxTime = sampler_t.sample(maxTime_s)

    # Размер области моделирования в отсчетах
    maxSize = sampler_x.sample(maxSize_m)

    # Положение источника в отсчетах
    sourcePos = sampler_x.sample(sourcePos_m)

    #Положение слоев в отсчетах
    layers = [sampleLayer(layer, sampler_x) for layer in layers_cont]

    # Датчик для регистрации поля
    probePos = sampler_x.sample(probePos_m)

    # Инициализация датчика
    probeEz = numpy.zeros(maxTime)
  
    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)

    # Магнитная проницаемость
    mu = numpy.ones(maxSize - 1)

    for layer in layers:
        fillMedium(layer, eps, mu)

    # Источник
    # Параметры гауссова сигнала
    A_0 = 100
    A_max = 100
    F_max = 3e9

    # Ширина и задержка гауссова сигнала
    w_g = numpy.sqrt(numpy.log(A_max)) / (numpy.pi * F_max)
    d_g = w_g * numpy.sqrt(numpy.log(A_0))

    source = Gaussian(1.0, d_g/dt, w_g/dt)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize - 1)

    # Массив, содержащий исходный импульс
    Ez0 = numpy.zeros(maxTime)

    # Создание экземпляров классов граничных условий
    boundary_left = boundary.ABCSecondLeft(eps[0], mu[0], Sc)
    boundary_right = boundary.ABCSecondRight(eps[-1], mu[-1], Sc)

    for t in range(1, maxTime):
        # Расчет компоненты поля H
        Hy = Hy + (Ez[1:] - Ez[:-1]) * Sc / (W0 * mu)

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -=(Sc / W0) * source.getField(t)

        # Расчет компоненты поля E
        Ez[1:-1] = Ez[1: -1] + (Hy[1:] - Hy[: -1]) * Sc * W0 / eps[1: -1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez0[t] += Sc * source.getField(t + 1.0)

        Ez[sourcePos] += Ez0[t]

        boundary_left.updateField(Ez, Hy)
        boundary_right.updateField(Ez, Hy)

        # Регистрация поля в датчике
        probeEz[t] = Ez[probePos]


    # Расчет спектра зарегистрированного сигнала
    Ez1Spec = fftshift(numpy.abs(fft(probeEz)))
    Ez0Spec = fftshift(numpy.abs(fft(Ez0)))
    Gamma = Ez1Spec / Ez0Spec

    # Шаг по частоте
    df = 1.0 / (maxTime * dt)

    # Расчет частоты
    freq = numpy.arange(-maxTime / 2 * df, maxTime / 2 * df, df)

    #Расчет области
    rang = numpy.arange(0, maxTime*dt, dt)

    # Отображение графиков
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    # Сигналы
    ax1.set_xlim(0, 0.15 * maxTime * dt)
    ax1.set_ylim(-0.6, 1.1)
    ax1.set_xlabel('t, с')
    ax1.set_ylabel('Ez, В/м')
    ax1.plot(rang, Ez0)
    ax1.plot(rang, probeEz)
    ax1.legend(['Падающий сигнал',
         'Отраженный сигнал'],
         loc='upper right')
    ax1.minorticks_on()
    ax1.grid()

    Fmax = 4e9
    Fmin = 1e9

    # Спектры сигналов
    ax2.set_xlim(0, 0.75 * Fmax)
    ax2.set_xlabel('f, Гц')
    ax2.set_ylabel('|F{Ez}|, В*с/м')
    ax2.plot(freq, Ez0Spec)
    ax2.plot(freq, Ez1Spec)
    ax2.legend(['Спектр падающего сигнала',
     'Спектр отраженного сигнала'],
     loc='upper right')
    ax2.minorticks_on()
    ax2.grid()

    # Коэффициент отражения
    ax3.set_xlim(Fmin, Fmax)
    ax3.set_ylim(0, 1.0)
    ax3.set_xlabel('f, Гц')
    ax3.set_ylabel('|Г|, б/р')
    ax3.plot(freq, Gamma)
    ax3.minorticks_on()
    ax3.grid()
    plt.subplots_adjust(hspace=0.5)
    plt.show()
