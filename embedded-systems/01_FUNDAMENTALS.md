# Embedded Systems Fundamentals

Programming for resource-constrained devices with real-time requirements.

---

## Table of Contents

1. [What Makes Embedded Different](#what-makes-embedded-different)
2. [Microcontroller Architecture](#microcontroller-architecture)
3. [Memory](#memory)
4. [Interrupts](#interrupts)
5. [Peripherals](#peripherals)
6. [Communication Protocols](#communication-protocols)
7. [Real-Time Concepts](#real-time-concepts)
8. [Power Management](#power-management)
9. [Common Microcontroller Families](#common-microcontroller-families)

---

## What Makes Embedded Different

| Desktop/Server | Embedded |
|----------------|----------|
| GB/TB of RAM | KB/MB of RAM |
| GHz multi-core CPUs | MHz single-core |
| Unlimited power | Battery/energy harvesting |
| OS handles everything | Bare metal or RTOS |
| Debugging is easy | Limited visibility |
| Failure = restart | Failure = crash/fire/injury |

**Key constraints:**
- **Memory**: Every byte counts
- **Speed**: Real-time deadlines
- **Power**: Battery life matters
- **Cost**: $0.10 difference × 1M units = $100K
- **Reliability**: May run for years without reboot

---

## Microcontroller Architecture

### Block Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      MICROCONTROLLER                             │
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │    CPU      │    │    Bus      │    │      Memory         │ │
│  │  (Core)     │◀──▶│  Matrix     │◀──▶│  Flash | SRAM | ROM │ │
│  └─────────────┘    └──────┬──────┘    └─────────────────────┘ │
│                            │                                    │
│         ┌──────────────────┼──────────────────┐                │
│         ▼                  ▼                  ▼                │
│  ┌────────────┐    ┌────────────┐    ┌────────────────┐       │
│  │   Timers   │    │    GPIO    │    │  Communication │       │
│  │            │    │            │    │  UART/SPI/I2C  │       │
│  └────────────┘    └────────────┘    └────────────────┘       │
│         │                  │                  │                │
│  ┌────────────┐    ┌────────────┐    ┌────────────────┐       │
│  │    ADC     │    │    DAC     │    │      DMA       │       │
│  └────────────┘    └────────────┘    └────────────────┘       │
│         │                  │                  │                │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          ▼                  ▼                  ▼
      Sensors            Actuators         External Memory
```

### CPU Core

Common architectures:

| Architecture | Examples | Use Cases |
|--------------|----------|-----------|
| **ARM Cortex-M** | STM32, nRF52, ESP32-S3 | General purpose, IoT |
| **ARM Cortex-A** | Raspberry Pi, BeagleBone | Linux-capable |
| **AVR** | Arduino Uno/Mega | Hobbyist, simple apps |
| **RISC-V** | ESP32-C3, SiFive | Open architecture |
| **Xtensa** | ESP32, ESP8266 | WiFi/BT applications |
| **PIC** | PIC16/18/32 | Industrial, automotive |

### Registers

Special memory locations that control hardware:

```c
// Direct register access (memory-mapped I/O)
#define GPIOA_ODR  (*(volatile uint32_t*)0x40020014)

GPIOA_ODR |= (1 << 5);   // Set bit 5 (turn on LED)
GPIOA_ODR &= ~(1 << 5);  // Clear bit 5 (turn off LED)
GPIOA_ODR ^= (1 << 5);   // Toggle bit 5

// Using vendor HAL (Hardware Abstraction Layer)
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
```

---

## Memory

### Memory Types

```
┌─────────────────────────────────────────────────────────────┐
│                     Memory Map                               │
│                                                              │
│   0xFFFF_FFFF ┌────────────────────┐                        │
│               │ Peripherals        │  GPIO, Timers, etc.    │
│               ├────────────────────┤                        │
│               │ Reserved           │                        │
│               ├────────────────────┤                        │
│   0x2000_0000 │ SRAM               │  Variables, stack      │
│               ├────────────────────┤                        │
│   0x0800_0000 │ Flash              │  Code, constants       │
│               ├────────────────────┤                        │
│   0x0000_0000 │ Boot ROM           │  Bootloader            │
│               └────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

| Memory Type | Persistent? | Speed | Use For |
|-------------|-------------|-------|---------|
| **Flash** | Yes | Slow to write | Code, constants |
| **SRAM** | No | Fast | Variables, stack, heap |
| **EEPROM** | Yes | Very slow | Configuration, calibration |
| **ROM** | Yes (fixed) | Fast | Bootloader |

### Memory Sections

```c
// .text - code (in Flash)
void myFunction(void) { ... }

// .rodata - read-only data (in Flash)
const char message[] = "Hello";
const int lookup_table[] = {1, 2, 3, 4};

// .data - initialized variables (copied from Flash to RAM at startup)
int counter = 42;

// .bss - uninitialized variables (zero-initialized in RAM)
int buffer[100];

// Stack (in RAM, grows down)
void func(void) {
    int local_var;  // on stack
}

// Heap (in RAM, grows up) - often avoided in embedded
int *p = malloc(100);  // dangerous in embedded!
```

### Memory-Conscious Programming

```c
// ❌ Wasteful
char buffer[1024];          // uses 1KB RAM always
float sensor_values[100];   // 400 bytes

// ✓ Memory efficient
char buffer[64];            // size appropriately
int16_t sensor_values[100]; // 200 bytes if range allows

// Use const for data that doesn't change (stays in Flash)
const uint8_t sine_table[256] = { ... };

// Use static for persistent local variables (no stack usage)
void updateFilter(int sample) {
    static int history[4];  // persists between calls
}

// Pack structures to save RAM
struct __attribute__((packed)) SensorData {
    uint8_t id;       // 1 byte
    uint16_t value;   // 2 bytes
    uint8_t status;   // 1 byte
};  // Total: 4 bytes (vs 8 with padding)
```

### Stack and Heap

```
RAM Layout:
┌─────────────────┐ High address
│     Stack       │ ← grows DOWN
│       ↓         │
│                 │
│       ↑         │
│     Heap        │ ← grows UP (if used)
├─────────────────┤
│     .bss        │ uninitialized globals
├─────────────────┤
│     .data       │ initialized globals
└─────────────────┘ Low address
```

**Stack overflow** is a common embedded bug - no MMU to catch it!

```c
// Check stack usage
#define STACK_CANARY 0xDEADBEEF

void checkStackOverflow(void) {
    extern uint32_t _stack_bottom;
    if (_stack_bottom != STACK_CANARY) {
        // Stack overflow detected!
        handleError();
    }
}
```

---

## Interrupts

### What is an Interrupt?

Hardware signal that pauses normal execution to handle an event.

```
Normal execution:      With interrupt:

   main()                 main()
     │                      │
     ▼                      ▼
   loop                   loop
     │                      │
     ▼                      │◀──── Button pressed!
   process               ┌─────────────────────┐
     │                   │   Save context      │
     ▼                   │   Run ISR           │
   continue              │   Restore context   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                                  continue
```

### Interrupt Service Routine (ISR)

```c
// ISR should be:
// - Fast (get in, get out)
// - No blocking (no delays, no waiting)
// - Minimal processing (set flag, let main loop handle)

volatile uint8_t button_pressed = 0;  // volatile is critical!

void EXTI0_IRQHandler(void) {
    if (EXTI->PR & (1 << 0)) {
        button_pressed = 1;        // Just set a flag
        EXTI->PR = (1 << 0);       // Clear interrupt flag
    }
}

int main(void) {
    while (1) {
        if (button_pressed) {
            button_pressed = 0;
            handleButtonPress();   // Heavy processing here
        }
    }
}
```

### Interrupt Priority

Most MCUs support priority levels:

```
Higher priority can interrupt lower priority (nesting)

Priority 0: Highest (system critical)
Priority 1: High (timing critical)
Priority 2: Medium (communication)
Priority 3: Low (background tasks)
```

### Common Interrupt Sources

| Source | Use Case |
|--------|----------|
| **GPIO/EXTI** | Button press, sensor signal |
| **Timer** | Periodic tasks, PWM, timing |
| **UART RX** | Serial data received |
| **ADC** | Conversion complete |
| **DMA** | Transfer complete |
| **SysTick** | OS tick, delays |

### Critical Sections

Protect shared data from interrupt corruption:

```c
volatile int shared_counter;

void incrementSafely(void) {
    __disable_irq();         // Disable interrupts
    shared_counter++;        // Critical section
    __enable_irq();          // Re-enable interrupts
}

// Or save/restore interrupt state
void criticalOperation(void) {
    uint32_t primask = __get_PRIMASK();
    __disable_irq();

    // Critical section
    doSomething();

    __set_PRIMASK(primask);  // Restore previous state
}
```

---

## Peripherals

### GPIO (General Purpose I/O)

```c
// Configure pin as output
GPIO_InitTypeDef gpio = {0};
gpio.Pin = GPIO_PIN_5;
gpio.Mode = GPIO_MODE_OUTPUT_PP;  // Push-pull output
gpio.Speed = GPIO_SPEED_LOW;
HAL_GPIO_Init(GPIOA, &gpio);

// Configure pin as input with pull-up
gpio.Pin = GPIO_PIN_0;
gpio.Mode = GPIO_MODE_INPUT;
gpio.Pull = GPIO_PULLUP;
HAL_GPIO_Init(GPIOA, &gpio);

// Read/write
HAL_GPIO_WritePin(GPIOA, GPIO_PIN_5, GPIO_PIN_SET);
int state = HAL_GPIO_ReadPin(GPIOA, GPIO_PIN_0);
```

### Timers

Versatile peripherals for timing, counting, PWM:

```c
// Basic timer interrupt (1ms tick)
TIM_HandleTypeDef htim;
htim.Instance = TIM2;
htim.Init.Prescaler = 84 - 1;        // 84MHz / 84 = 1MHz
htim.Init.Period = 1000 - 1;         // 1MHz / 1000 = 1kHz = 1ms
HAL_TIM_Base_Init(&htim);
HAL_TIM_Base_Start_IT(&htim);

void TIM2_IRQHandler(void) {
    HAL_TIM_IRQHandler(&htim);
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
    // Called every 1ms
    milliseconds++;
}
```

### PWM (Pulse Width Modulation)

Control motors, LEDs, servos:

```
PWM Signal:
     ┌───┐     ┌───┐     ┌───┐
     │   │     │   │     │   │
─────┘   └─────┘   └─────┘   └─────

Duty Cycle = ON time / Period
50% duty = half power/brightness
```

```c
// Configure PWM on TIM3 Channel 1
htim.Init.Period = 1000 - 1;  // 1000 steps resolution
HAL_TIM_PWM_Init(&htim);
HAL_TIM_PWM_Start(&htim, TIM_CHANNEL_1);

// Set duty cycle (0-100%)
__HAL_TIM_SET_COMPARE(&htim, TIM_CHANNEL_1, 500);  // 50%
```

### ADC (Analog to Digital Converter)

Read analog sensors:

```c
// Single conversion
HAL_ADC_Start(&hadc);
HAL_ADC_PollForConversion(&hadc, 100);
uint32_t raw = HAL_ADC_GetValue(&hadc);

// Convert to voltage (for 12-bit ADC, 3.3V reference)
float voltage = (raw / 4095.0f) * 3.3f;

// Convert to temperature (example sensor)
float temp_c = (voltage - 0.5f) * 100.0f;
```

### DMA (Direct Memory Access)

Transfer data without CPU intervention:

```c
// DMA transfer from ADC to buffer
uint16_t adc_buffer[100];
HAL_ADC_Start_DMA(&hadc, (uint32_t*)adc_buffer, 100);

void HAL_ADC_ConvCpltCallback(ADC_HandleTypeDef *hadc) {
    // Buffer is full, process data
    processADCData(adc_buffer, 100);
}
```

---

## Communication Protocols

### UART (Serial)

Asynchronous, point-to-point:

```
TX ─────────────────────────────────────────── RX
RX ─────────────────────────────────────────── TX
GND ────────────────────────────────────────── GND

Frame: [START][D0][D1][D2][D3][D4][D5][D6][D7][PARITY][STOP]

Common baud rates: 9600, 115200, 921600
```

```c
// Transmit
HAL_UART_Transmit(&huart, data, length, timeout);

// Receive with interrupt
HAL_UART_Receive_IT(&huart, rx_buffer, 1);

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart) {
    processReceivedByte(rx_buffer[0]);
    HAL_UART_Receive_IT(huart, rx_buffer, 1);  // Re-arm
}
```

### SPI (Serial Peripheral Interface)

Synchronous, full-duplex, multi-device:

```
        Master              Slave
        ┌────┐              ┌────┐
  MOSI  │    │─────────────▶│    │  Master Out, Slave In
  MISO  │    │◀─────────────│    │  Master In, Slave Out
  SCK   │    │─────────────▶│    │  Clock (master controls)
  CS    │    │─────────────▶│    │  Chip Select (active low)
        └────┘              └────┘

Speed: Up to 100+ MHz
```

```c
// SPI transaction
uint8_t tx_data[] = {0x01, 0x02};
uint8_t rx_data[2];

HAL_GPIO_WritePin(CS_PORT, CS_PIN, GPIO_PIN_RESET);  // Select
HAL_SPI_TransmitReceive(&hspi, tx_data, rx_data, 2, 100);
HAL_GPIO_WritePin(CS_PORT, CS_PIN, GPIO_PIN_SET);    // Deselect
```

### I2C (Inter-Integrated Circuit)

Synchronous, multi-master, multi-slave, 2 wires:

```
        Master                Slave 1           Slave 2
        ┌────┐                ┌────┐            ┌────┐
  SDA   │    │────────────────│    │────────────│    │  Data
  SCL   │    │────────────────│    │────────────│    │  Clock
        └────┘                └────┘            └────┘

Each device has unique 7-bit address
Speed: 100kHz (standard), 400kHz (fast), 1MHz (fast+)
```

```c
// Write to device
uint8_t data[] = {REGISTER_ADDR, value};
HAL_I2C_Master_Transmit(&hi2c, DEVICE_ADDR << 1, data, 2, 100);

// Read from device
HAL_I2C_Master_Transmit(&hi2c, DEVICE_ADDR << 1, &reg, 1, 100);
HAL_I2C_Master_Receive(&hi2c, DEVICE_ADDR << 1, buffer, len, 100);

// Or combined write-then-read
HAL_I2C_Mem_Read(&hi2c, DEVICE_ADDR << 1, reg, I2C_MEMADD_SIZE_8BIT,
                 buffer, len, 100);
```

### Protocol Comparison

| Protocol | Wires | Speed | Distance | Devices | Use Case |
|----------|-------|-------|----------|---------|----------|
| **UART** | 2 | ~1 Mbps | Long | 2 | Debug, GPS, BT modules |
| **SPI** | 4+ | 100 MHz | Short | Few | Flash, displays, ADCs |
| **I2C** | 2 | 1 MHz | Short | Many | Sensors, EEPROMs |
| **CAN** | 2 | 1 Mbps | Long | Many | Automotive, industrial |

---

## Real-Time Concepts

### What is Real-Time?

**Real-time ≠ fast.** Real-time = **predictable, meeting deadlines.**

| Type | Definition | Example |
|------|------------|---------|
| **Hard real-time** | Missing deadline = failure | Airbag, pacemaker |
| **Firm real-time** | Missing deadline = degraded | Video frame drop |
| **Soft real-time** | Missing deadline = annoying | UI responsiveness |

### Bare Metal vs RTOS

**Bare Metal (Super Loop):**
```c
int main(void) {
    init();
    while (1) {
        readSensors();
        processData();
        updateOutputs();
        communicate();
    }
}

// Simple, but:
// - Hard to meet timing requirements
// - One slow task blocks everything
// - No prioritization
```

**RTOS (Real-Time Operating System):**
```c
void sensorTask(void *arg) {
    while (1) {
        readSensors();
        vTaskDelay(pdMS_TO_TICKS(10));  // Run every 10ms
    }
}

void commTask(void *arg) {
    while (1) {
        sendData();
        vTaskDelay(pdMS_TO_TICKS(100));  // Run every 100ms
    }
}

int main(void) {
    xTaskCreate(sensorTask, "Sensor", 128, NULL, HIGH_PRIORITY, NULL);
    xTaskCreate(commTask, "Comm", 256, NULL, LOW_PRIORITY, NULL);
    vTaskStartScheduler();
}
```

### Common RTOS Concepts

**Tasks/Threads:** Independent execution units
**Semaphores:** Synchronization, resource counting
**Mutexes:** Mutual exclusion for shared resources
**Queues:** Safe data passing between tasks
**Events/Signals:** Task notification

### Timing Analysis

```
Worst-Case Execution Time (WCET):
- Measure maximum time for each task
- Ensure sum fits within period

Example:
Task A: 2ms WCET, 10ms period
Task B: 5ms WCET, 20ms period
Task C: 1ms WCET, 5ms period

CPU utilization = 2/10 + 5/20 + 1/5 = 0.2 + 0.25 + 0.2 = 65%
(Should keep below 70-80% for safety margin)
```

---

## Power Management

### Power Modes

```
Run Mode:        CPU active, all peripherals available
                 Power: 100%

Sleep Mode:      CPU stopped, peripherals running
                 Wake: Any interrupt
                 Power: ~50%

Stop Mode:       CPU + most peripherals stopped
                 Wake: External interrupt, RTC
                 Power: ~5%

Standby Mode:    Everything off except RTC + backup
                 Wake: External pin, RTC
                 Power: ~0.1%
```

### Power Optimization Techniques

```c
// 1. Use lowest clock speed sufficient
RCC_ClkInitTypeDef clk = {0};
clk.ClockType = RCC_CLOCKTYPE_SYSCLK;
clk.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;  // 16MHz instead of 168MHz
HAL_RCC_ClockConfig(&clk, FLASH_LATENCY_0);

// 2. Disable unused peripherals
__HAL_RCC_GPIOB_CLK_DISABLE();
__HAL_RCC_SPI1_CLK_DISABLE();

// 3. Use sleep mode when idle
while (1) {
    if (!work_pending) {
        __WFI();  // Wait For Interrupt (sleep until interrupt)
    }
    processWork();
}

// 4. Batch operations
// Instead of: wake → send 1 byte → sleep (repeated)
// Do: wake → send 100 bytes → sleep
```

### Battery Life Estimation

```
Battery capacity: 1000 mAh

Active mode: 50mA for 10ms every second
Sleep mode: 0.01mA rest of time

Average current = (50mA × 10ms + 0.01mA × 990ms) / 1000ms
                = (0.5 + 0.0099) / 1 = 0.51 mA

Battery life = 1000mAh / 0.51mA = ~2000 hours = 83 days
```

---

## Common Microcontroller Families

### Comparison

| Family | Strengths | Best For |
|--------|-----------|----------|
| **STM32** | Wide range, great docs, CubeMX | Professional development |
| **ESP32** | WiFi/BT built-in, cheap | IoT projects |
| **nRF52** | BLE, low power | Wearables, sensors |
| **Arduino (AVR)** | Easy to start | Learning, prototypes |
| **RP2040** | PIO, cheap, dual-core | Custom protocols |
| **Teensy** | Fast, Arduino-compatible | Audio, high-speed |

### Development Boards

| Board | MCU | Features | Price |
|-------|-----|----------|-------|
| **STM32 Nucleo** | STM32 | ST-Link debugger, Arduino headers | $10-20 |
| **ESP32 DevKit** | ESP32 | WiFi, BT, lots of GPIO | $5-10 |
| **Arduino Uno** | ATmega328P | 5V, simple | $10-25 |
| **Raspberry Pi Pico** | RP2040 | Dual-core, PIO | $4 |
| **nRF52 DK** | nRF52840 | BLE, debugging | $40 |

---

## Quick Reference

### Bit Manipulation

```c
#define BIT(n)          (1U << (n))
#define SET_BIT(x, n)   ((x) |= BIT(n))
#define CLR_BIT(x, n)   ((x) &= ~BIT(n))
#define TGL_BIT(x, n)   ((x) ^= BIT(n))
#define GET_BIT(x, n)   (((x) >> (n)) & 1)

// Set bits 3-5
REG |= (0x7 << 3);

// Clear bits 3-5
REG &= ~(0x7 << 3);

// Read bits 3-5
value = (REG >> 3) & 0x7;
```

### Common Macros

```c
#define ARRAY_SIZE(arr)     (sizeof(arr) / sizeof((arr)[0]))
#define MIN(a, b)           (((a) < (b)) ? (a) : (b))
#define MAX(a, b)           (((a) > (b)) ? (a) : (b))
#define CLAMP(x, lo, hi)    (MIN(MAX(x, lo), hi))

// Memory barrier (prevent reordering)
#define MEMORY_BARRIER()    __asm__ __volatile__("" ::: "memory")
```

### Volatile Keyword

**Always use `volatile` for:**
- Hardware registers
- Variables modified by ISR
- Variables shared between tasks

```c
volatile uint32_t *REG = (volatile uint32_t *)0x40000000;
volatile int isr_flag = 0;
```

---

Next: [Practical Embedded Systems](./02_PRACTICAL.md) - Debugging, patterns, and real-world tips
