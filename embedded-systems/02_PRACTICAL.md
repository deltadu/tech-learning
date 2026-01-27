# Practical Embedded Systems

Debugging techniques, common patterns, and real-world tips.

---

## Table of Contents

1. [Development Workflow](#development-workflow)
2. [Debugging Techniques](#debugging-techniques)
3. [Common Patterns](#common-patterns)
4. [State Machines](#state-machines)
5. [Circular Buffers](#circular-buffers)
6. [Fixed-Point Math](#fixed-point-math)
7. [Watchdog & Error Handling](#watchdog--error-handling)
8. [Testing Embedded Code](#testing-embedded-code)
9. [FreeRTOS Basics](#freertos-basics)
10. [Common Gotchas](#common-gotchas)

---

## Development Workflow

### Toolchain

```
┌────────────────────────────────────────────────────────────────┐
│ Development Workflow                                            │
│                                                                 │
│  Source Code (.c/.h)                                            │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────┐                                               │
│  │  Compiler    │  arm-none-eabi-gcc                           │
│  │  (Cross)     │                                               │
│  └──────┬───────┘                                               │
│         │                                                       │
│         ▼                                                       │
│  Object Files (.o)                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │   Linker     │◀────│ Linker Script│  Memory layout          │
│  └──────┬───────┘     └──────────────┘                         │
│         │                                                       │
│         ▼                                                       │
│  Executable (.elf)                                              │
│         │                                                       │
│         ├──────────────▶ .bin/.hex (for flashing)              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐     ┌──────────────┐                         │
│  │   Debugger   │◀───▶│ Debug Probe  │  ST-Link, J-Link        │
│  │   (GDB)      │     │              │                         │
│  └──────────────┘     └──────────────┘                         │
│                              │                                  │
│                              ▼                                  │
│                       ┌──────────────┐                         │
│                       │    Target    │                         │
│                       │    MCU       │                         │
│                       └──────────────┘                         │
└────────────────────────────────────────────────────────────────┘
```

### Build System

**Makefile basics:**
```makefile
CC = arm-none-eabi-gcc
CFLAGS = -mcpu=cortex-m4 -mthumb -O2 -g
LDFLAGS = -T linker.ld -nostartfiles

SRC = main.c startup.c
OBJ = $(SRC:.c=.o)

firmware.elf: $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

flash: firmware.elf
	openocd -f interface/stlink.cfg -f target/stm32f4x.cfg \
		-c "program firmware.elf verify reset exit"

clean:
	rm -f $(OBJ) firmware.elf
```

**Or use CMake with toolchain file.**

---

## Debugging Techniques

### Printf Debugging

Simple but effective:

```c
// Redirect printf to UART
int _write(int fd, char *ptr, int len) {
    HAL_UART_Transmit(&huart2, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    return len;
}

// Usage
printf("Sensor: %d, State: %d\n", sensor_value, current_state);

// Lightweight alternative (no formatting overhead)
void debug_hex(uint32_t val) {
    char buf[11] = "0x00000000\n";
    for (int i = 9; i >= 2; i--) {
        int nibble = val & 0xF;
        buf[i] = nibble < 10 ? '0' + nibble : 'A' + nibble - 10;
        val >>= 4;
    }
    HAL_UART_Transmit(&huart, (uint8_t*)buf, 11, 100);
}
```

### LED Debugging

When UART isn't available:

```c
// Blink pattern indicates error code
void indicateError(int code) {
    while (1) {
        for (int i = 0; i < code; i++) {
            LED_ON();
            delay_ms(200);
            LED_OFF();
            delay_ms(200);
        }
        delay_ms(1000);  // Pause between patterns
    }
}

// Heartbeat LED (proves main loop is running)
void heartbeat(void) {
    static uint32_t last = 0;
    if (HAL_GetTick() - last > 500) {
        HAL_GPIO_TogglePin(LED_PORT, LED_PIN);
        last = HAL_GetTick();
    }
}
```

### Logic Analyzer / Oscilloscope

Toggle GPIO to measure timing:

```c
// Measure function execution time
GPIO_SET(DEBUG_PIN);
expensiveFunction();
GPIO_CLEAR(DEBUG_PIN);

// Measure on oscilloscope: pulse width = execution time
```

### Hardware Debugger (GDB)

```bash
# Connect to target
arm-none-eabi-gdb firmware.elf
(gdb) target remote :3333

# Common commands
(gdb) break main              # Set breakpoint
(gdb) continue                # Run
(gdb) step                    # Step into
(gdb) next                    # Step over
(gdb) print variable          # Print value
(gdb) x/10xw 0x20000000       # Examine memory
(gdb) info registers          # Show registers
(gdb) backtrace               # Call stack
```

### Fault Debugging

When hard fault occurs:

```c
void HardFault_Handler(void) {
    __asm volatile (
        "tst lr, #4          \n"
        "ite eq              \n"
        "mrseq r0, msp       \n"
        "mrsne r0, psp       \n"
        "b HardFault_Handler_C \n"
    );
}

void HardFault_Handler_C(uint32_t *stack) {
    volatile uint32_t r0 = stack[0];
    volatile uint32_t r1 = stack[1];
    volatile uint32_t r2 = stack[2];
    volatile uint32_t r3 = stack[3];
    volatile uint32_t r12 = stack[4];
    volatile uint32_t lr = stack[5];   // Return address
    volatile uint32_t pc = stack[6];   // Faulting instruction
    volatile uint32_t psr = stack[7];

    // Set breakpoint here and inspect values
    while (1);
}
```

---

## Common Patterns

### Debouncing

Eliminate button bounce (mechanical switches bounce for ~5-20ms):

```c
#define DEBOUNCE_MS 20

bool readButtonDebounced(void) {
    static uint32_t lastChange = 0;
    static bool lastState = false;
    static bool debouncedState = false;

    bool currentState = HAL_GPIO_ReadPin(BTN_PORT, BTN_PIN) == GPIO_PIN_SET;

    if (currentState != lastState) {
        lastChange = HAL_GetTick();
        lastState = currentState;
    }

    if ((HAL_GetTick() - lastChange) > DEBOUNCE_MS) {
        debouncedState = currentState;
    }

    return debouncedState;
}

// Or simpler: edge detection with debounce
bool getButtonPress(void) {
    static bool lastStable = false;
    bool current = readButtonDebounced();

    if (current && !lastStable) {
        lastStable = current;
        return true;  // Rising edge
    }
    lastStable = current;
    return false;
}
```

### Timing Without Blocking

Never use busy-wait delays in main loop:

```c
// ❌ Bad: blocks everything
void loop(void) {
    readSensor();
    HAL_Delay(100);  // Blocks for 100ms!
}

// ✓ Good: non-blocking timing
void loop(void) {
    static uint32_t lastSensorRead = 0;
    static uint32_t lastLedToggle = 0;

    uint32_t now = HAL_GetTick();

    if (now - lastSensorRead >= 100) {
        readSensor();
        lastSensorRead = now;
    }

    if (now - lastLedToggle >= 500) {
        toggleLed();
        lastLedToggle = now;
    }

    // Other tasks run without blocking
    processCommands();
}
```

### Software Timers

```c
typedef struct {
    uint32_t interval;
    uint32_t lastTick;
    void (*callback)(void);
} SoftTimer;

SoftTimer timers[MAX_TIMERS];

void timerInit(int id, uint32_t interval_ms, void (*cb)(void)) {
    timers[id].interval = interval_ms;
    timers[id].lastTick = HAL_GetTick();
    timers[id].callback = cb;
}

void timerProcess(void) {
    uint32_t now = HAL_GetTick();
    for (int i = 0; i < MAX_TIMERS; i++) {
        if (timers[i].callback &&
            (now - timers[i].lastTick >= timers[i].interval)) {
            timers[i].lastTick = now;
            timers[i].callback();
        }
    }
}

// Usage
void onSensorTimer(void) { readSensors(); }
void onHeartbeat(void) { toggleLed(); }

int main(void) {
    timerInit(0, 100, onSensorTimer);
    timerInit(1, 500, onHeartbeat);

    while (1) {
        timerProcess();
    }
}
```

### Moving Average Filter

Smooth noisy sensor readings:

```c
#define FILTER_SIZE 8

typedef struct {
    int16_t buffer[FILTER_SIZE];
    uint8_t index;
    int32_t sum;
} MovingAverage;

void maInit(MovingAverage *ma) {
    memset(ma->buffer, 0, sizeof(ma->buffer));
    ma->index = 0;
    ma->sum = 0;
}

int16_t maUpdate(MovingAverage *ma, int16_t sample) {
    ma->sum -= ma->buffer[ma->index];
    ma->buffer[ma->index] = sample;
    ma->sum += sample;
    ma->index = (ma->index + 1) % FILTER_SIZE;
    return ma->sum / FILTER_SIZE;
}

// Usage
MovingAverage tempFilter;
maInit(&tempFilter);

while (1) {
    int16_t raw = readADC();
    int16_t filtered = maUpdate(&tempFilter, raw);
}
```

---

## State Machines

Organize complex behavior:

```c
typedef enum {
    STATE_IDLE,
    STATE_RUNNING,
    STATE_ERROR,
    STATE_COUNT
} State;

typedef enum {
    EVENT_START,
    EVENT_STOP,
    EVENT_ERROR,
    EVENT_RESET,
    EVENT_COUNT
} Event;

typedef struct {
    State nextState;
    void (*action)(void);
} Transition;

// Transition table
Transition transitions[STATE_COUNT][EVENT_COUNT] = {
    // STATE_IDLE
    [STATE_IDLE] = {
        [EVENT_START] = {STATE_RUNNING, onStart},
        [EVENT_STOP]  = {STATE_IDLE, NULL},
        [EVENT_ERROR] = {STATE_ERROR, onError},
        [EVENT_RESET] = {STATE_IDLE, NULL},
    },
    // STATE_RUNNING
    [STATE_RUNNING] = {
        [EVENT_START] = {STATE_RUNNING, NULL},
        [EVENT_STOP]  = {STATE_IDLE, onStop},
        [EVENT_ERROR] = {STATE_ERROR, onError},
        [EVENT_RESET] = {STATE_IDLE, onReset},
    },
    // STATE_ERROR
    [STATE_ERROR] = {
        [EVENT_START] = {STATE_ERROR, NULL},
        [EVENT_STOP]  = {STATE_ERROR, NULL},
        [EVENT_ERROR] = {STATE_ERROR, NULL},
        [EVENT_RESET] = {STATE_IDLE, onReset},
    },
};

State currentState = STATE_IDLE;

void processEvent(Event event) {
    Transition *t = &transitions[currentState][event];
    if (t->action) {
        t->action();
    }
    currentState = t->nextState;
}
```

---

## Circular Buffers

Safe producer-consumer pattern:

```c
#define BUFFER_SIZE 256  // Must be power of 2

typedef struct {
    uint8_t data[BUFFER_SIZE];
    volatile uint16_t head;  // Write index
    volatile uint16_t tail;  // Read index
} CircularBuffer;

void cbInit(CircularBuffer *cb) {
    cb->head = 0;
    cb->tail = 0;
}

bool cbIsFull(CircularBuffer *cb) {
    return ((cb->head + 1) & (BUFFER_SIZE - 1)) == cb->tail;
}

bool cbIsEmpty(CircularBuffer *cb) {
    return cb->head == cb->tail;
}

bool cbWrite(CircularBuffer *cb, uint8_t byte) {
    if (cbIsFull(cb)) return false;
    cb->data[cb->head] = byte;
    cb->head = (cb->head + 1) & (BUFFER_SIZE - 1);
    return true;
}

bool cbRead(CircularBuffer *cb, uint8_t *byte) {
    if (cbIsEmpty(cb)) return false;
    *byte = cb->data[cb->tail];
    cb->tail = (cb->tail + 1) & (BUFFER_SIZE - 1);
    return true;
}

uint16_t cbCount(CircularBuffer *cb) {
    return (cb->head - cb->tail) & (BUFFER_SIZE - 1);
}

// Usage: ISR writes, main loop reads
CircularBuffer uartRxBuffer;

void UART_IRQHandler(void) {
    uint8_t byte = UART->DR;
    cbWrite(&uartRxBuffer, byte);  // ISR produces
}

void processUART(void) {
    uint8_t byte;
    while (cbRead(&uartRxBuffer, &byte)) {  // Main loop consumes
        handleByte(byte);
    }
}
```

---

## Fixed-Point Math

When float is too slow/unavailable:

```c
// Q16.16 fixed point (16 bits integer, 16 bits fraction)
typedef int32_t fixed_t;

#define FIXED_SHIFT     16
#define FIXED_ONE       (1 << FIXED_SHIFT)
#define FLOAT_TO_FIXED(f)   ((fixed_t)((f) * FIXED_ONE))
#define FIXED_TO_FLOAT(f)   ((float)(f) / FIXED_ONE)
#define INT_TO_FIXED(i)     ((fixed_t)(i) << FIXED_SHIFT)
#define FIXED_TO_INT(f)     ((f) >> FIXED_SHIFT)

// Arithmetic
fixed_t fixed_add(fixed_t a, fixed_t b) { return a + b; }
fixed_t fixed_sub(fixed_t a, fixed_t b) { return a - b; }

fixed_t fixed_mul(fixed_t a, fixed_t b) {
    return (fixed_t)(((int64_t)a * b) >> FIXED_SHIFT);
}

fixed_t fixed_div(fixed_t a, fixed_t b) {
    return (fixed_t)(((int64_t)a << FIXED_SHIFT) / b);
}

// Example: PID controller
typedef struct {
    fixed_t kp, ki, kd;
    fixed_t integral;
    fixed_t lastError;
} PIDController;

fixed_t pidUpdate(PIDController *pid, fixed_t error, fixed_t dt) {
    pid->integral = fixed_add(pid->integral, fixed_mul(error, dt));
    fixed_t derivative = fixed_div(fixed_sub(error, pid->lastError), dt);
    pid->lastError = error;

    return fixed_add(
        fixed_add(
            fixed_mul(pid->kp, error),
            fixed_mul(pid->ki, pid->integral)
        ),
        fixed_mul(pid->kd, derivative)
    );
}
```

---

## Watchdog & Error Handling

### Watchdog Timer

Reset MCU if software hangs:

```c
// Initialize watchdog (reset if not fed within timeout)
void watchdogInit(uint32_t timeout_ms) {
    IWDG->KR = 0x5555;  // Enable register access
    IWDG->PR = 4;       // Prescaler
    IWDG->RLR = timeout_ms * 40 / 256;  // Reload value
    IWDG->KR = 0xCCCC;  // Start watchdog
}

void watchdogFeed(void) {
    IWDG->KR = 0xAAAA;  // Reset counter
}

int main(void) {
    watchdogInit(1000);  // 1 second timeout

    while (1) {
        doWork();
        watchdogFeed();  // Must call regularly
    }
}
```

### Defensive Programming

```c
// Assert for development
#ifdef DEBUG
#define ASSERT(cond) do { \
    if (!(cond)) { \
        printf("ASSERT: %s:%d\n", __FILE__, __LINE__); \
        while(1); \
    } \
} while(0)
#else
#define ASSERT(cond)
#endif

// Parameter validation
int readSensor(int channel) {
    ASSERT(channel >= 0 && channel < NUM_CHANNELS);
    if (channel < 0 || channel >= NUM_CHANNELS) {
        return ERROR_INVALID_PARAM;
    }
    // ...
}

// Null pointer checks
void processData(uint8_t *data, size_t len) {
    if (data == NULL || len == 0) {
        return;
    }
    // ...
}

// Timeout on operations
bool waitForReady(uint32_t timeout_ms) {
    uint32_t start = HAL_GetTick();
    while (!isReady()) {
        if (HAL_GetTick() - start > timeout_ms) {
            return false;  // Timeout
        }
    }
    return true;
}
```

### Error Codes

```c
typedef enum {
    ERR_OK = 0,
    ERR_TIMEOUT,
    ERR_INVALID_PARAM,
    ERR_HARDWARE,
    ERR_BUSY,
    ERR_NO_MEMORY,
} ErrorCode;

const char *errorToString(ErrorCode err) {
    static const char *names[] = {
        "OK", "TIMEOUT", "INVALID_PARAM", "HARDWARE", "BUSY", "NO_MEMORY"
    };
    return err < sizeof(names)/sizeof(names[0]) ? names[err] : "UNKNOWN";
}

// Global error tracking
volatile ErrorCode lastError = ERR_OK;
volatile uint32_t errorCount = 0;

void setError(ErrorCode err) {
    lastError = err;
    errorCount++;
    // Log, blink LED, etc.
}
```

---

## Testing Embedded Code

### Unit Testing (on host)

Test logic on PC before deploying:

```c
// sensor_logic.c - platform-independent logic
int convertRawToTemp(int raw) {
    return (raw * 330 - 50000) / 1000;  // Example conversion
}

// test_sensor.c - run on PC
#include <assert.h>

void test_convertRawToTemp(void) {
    assert(convertRawToTemp(0) == -50);
    assert(convertRawToTemp(1000) == 280);
    assert(convertRawToTemp(500) == 115);
    printf("All tests passed!\n");
}

int main(void) {
    test_convertRawToTemp();
    return 0;
}
```

### Hardware Abstraction

```c
// hal.h - abstract interface
typedef struct {
    void (*gpioWrite)(int pin, int value);
    int (*gpioRead)(int pin);
    uint32_t (*getTick)(void);
} HAL;

// hal_stm32.c - real implementation
void stm32_gpioWrite(int pin, int value) {
    HAL_GPIO_WritePin(GPIOA, pin, value);
}
HAL hal = { stm32_gpioWrite, stm32_gpioRead, HAL_GetTick };

// hal_mock.c - for testing
int mockPinState[32];
void mock_gpioWrite(int pin, int value) { mockPinState[pin] = value; }
HAL hal = { mock_gpioWrite, mock_gpioRead, mock_getTick };

// Application code uses HAL
void blinkLed(HAL *hal) {
    hal->gpioWrite(LED_PIN, 1);
    // ...
}
```

---

## FreeRTOS Basics

### Task Creation

```c
#include "FreeRTOS.h"
#include "task.h"

void sensorTask(void *params) {
    while (1) {
        int value = readSensor();
        processValue(value);
        vTaskDelay(pdMS_TO_TICKS(100));  // Delay 100ms
    }
}

void commTask(void *params) {
    while (1) {
        sendData();
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}

int main(void) {
    xTaskCreate(sensorTask, "Sensor", 256, NULL, 2, NULL);
    xTaskCreate(commTask, "Comm", 512, NULL, 1, NULL);
    vTaskStartScheduler();

    while (1);  // Should never reach here
}
```

### Queues

Safe data passing between tasks:

```c
#include "queue.h"

typedef struct {
    int sensor_id;
    int value;
} SensorReading;

QueueHandle_t sensorQueue;

void producerTask(void *params) {
    while (1) {
        SensorReading reading = {1, readSensor()};
        xQueueSend(sensorQueue, &reading, portMAX_DELAY);
        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

void consumerTask(void *params) {
    SensorReading reading;
    while (1) {
        if (xQueueReceive(sensorQueue, &reading, portMAX_DELAY)) {
            printf("Sensor %d: %d\n", reading.sensor_id, reading.value);
        }
    }
}

int main(void) {
    sensorQueue = xQueueCreate(10, sizeof(SensorReading));
    // Create tasks...
}
```

### Mutexes

Protect shared resources:

```c
#include "semphr.h"

SemaphoreHandle_t uartMutex;

void safePrint(const char *msg) {
    xSemaphoreTake(uartMutex, portMAX_DELAY);
    printf("%s", msg);
    xSemaphoreGive(uartMutex);
}

int main(void) {
    uartMutex = xSemaphoreCreateMutex();
    // Create tasks...
}
```

---

## Common Gotchas

### 1. Forgetting volatile

```c
// ❌ Optimizer might cache value
int flag = 0;
void ISR(void) { flag = 1; }
while (!flag);  // May infinite loop!

// ✓ Tells compiler variable can change unexpectedly
volatile int flag = 0;
```

### 2. Integer Overflow

```c
// ❌ Overflow before cast
uint32_t result = adc_value * 3300 / 4095;  // Overflow if adc > 1302!

// ✓ Cast first
uint32_t result = (uint32_t)adc_value * 3300 / 4095;
```

### 3. Unaligned Access

```c
// ❌ May crash on ARM
uint32_t *ptr = (uint32_t*)(buffer + 1);  // Unaligned!
uint32_t value = *ptr;

// ✓ Use memcpy for unaligned data
uint32_t value;
memcpy(&value, buffer + 1, sizeof(value));
```

### 4. Stack Overflow

```c
// ❌ Large local arrays
void badFunction(void) {
    uint8_t buffer[4096];  // Stack overflow!
}

// ✓ Use static or heap
static uint8_t buffer[4096];
void goodFunction(void) {
    // Use buffer...
}
```

### 5. Interrupt Priority Inversion

```c
// ❌ Low-priority ISR takes mutex, high-priority can't get it
// Use proper RTOS primitives or disable interrupts briefly
```

### 6. Endianness

```c
// Network byte order is big-endian, ARM is little-endian
uint16_t value = 0x1234;

// ❌ Sending raw bytes
send(&value, 2);  // Sends 0x34, 0x12 on little-endian

// ✓ Convert to network order
uint16_t net_value = htons(value);  // Host to network short
send(&net_value, 2);  // Sends 0x12, 0x34
```

---

## Quick Reference

### Common Conversions

```c
// ADC to voltage (12-bit, 3.3V reference)
float voltage = (float)adc_raw * 3.3f / 4095.0f;

// Voltage to temperature (typical sensor)
float temp_c = (voltage - 0.5f) * 100.0f;

// PWM duty cycle (0-100%)
uint32_t pwm_value = (duty_percent * period) / 100;

// Timer period for frequency
uint32_t period = timer_clock / desired_frequency - 1;
```

### Memory Sizes

```c
sizeof(char)      = 1
sizeof(int16_t)   = 2
sizeof(int32_t)   = 4
sizeof(int64_t)   = 8
sizeof(float)     = 4
sizeof(double)    = 8
sizeof(void*)     = 4 (32-bit MCU)
```

### Startup Checklist

```
1. ✓ System clock configured
2. ✓ Peripheral clocks enabled
3. ✓ GPIO initialized
4. ✓ Interrupts configured and enabled
5. ✓ Watchdog initialized
6. ✓ Communication peripherals ready
7. ✓ Error handlers in place
```
