################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (11.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../AD5940Lib/Src/NUCLEOF401Port.c \
../AD5940Lib/Src/ad5940.c 

OBJS += \
./AD5940Lib/Src/NUCLEOF401Port.o \
./AD5940Lib/Src/ad5940.o 

C_DEPS += \
./AD5940Lib/Src/NUCLEOF401Port.d \
./AD5940Lib/Src/ad5940.d 


# Each subdirectory must supply rules for building sources it contributes
AD5940Lib/Src/%.o AD5940Lib/Src/%.su AD5940Lib/Src/%.cyclo: ../AD5940Lib/Src/%.c AD5940Lib/Src/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F401xE -c -I../Core/Inc -I../AD5940Lib/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-AD5940Lib-2f-Src

clean-AD5940Lib-2f-Src:
	-$(RM) ./AD5940Lib/Src/NUCLEOF401Port.cyclo ./AD5940Lib/Src/NUCLEOF401Port.d ./AD5940Lib/Src/NUCLEOF401Port.o ./AD5940Lib/Src/NUCLEOF401Port.su ./AD5940Lib/Src/ad5940.cyclo ./AD5940Lib/Src/ad5940.d ./AD5940Lib/Src/ad5940.o ./AD5940Lib/Src/ad5940.su

.PHONY: clean-AD5940Lib-2f-Src

