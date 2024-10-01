/*

Copyright (c) 2017-2019 Analog Devices, Inc. All Rights Reserved.

This software is proprietary to Analog Devices, Inc. and its licensors.
By using this software you agree to the terms of the associated
Analog Devices Software License Agreement.

*/
#include "stdio.h"
#include "ad5940.h"
#include "stm32f4xx_hal.h"
#include "stm32f4xx_it.h"
#include "main.h"
#include "math.h"

UART_HandleTypeDef UartHandle;
DMA_HandleTypeDef hdma_usart2_rx;
static const char *const FW_Date = __DATE__;
static const char *const FW_Time = __TIME__;

//static const long delay_mesure = 60000; //delay entre deux mesure en milliseconde



/* Functions that used to initialize MCU platform */
void MCUPlatformInit(void *pCfg);
static void MX_DMA_Init(void);
//static void MX_USART2_UART_Init(void);


int fini = 0;


int main(void)
{
  //static int32_t AD5940PlatformCfg(void);
  void AD5940_BIA(long delay_mesure, uint32_t nb_pt, float freq_min, float freq_max, long nb_mesure);
  void serial_reading(long *delay_mesure, uint32_t *nb_pt, float *freq_min, float *freq_max, long *nb_mesure);

  float freq_min;		//freq min du balayage
  float freq_max;		//freq max du balayage
  uint32_t nb_pt;		//nb de point durant le balayage
  long delay_mesure;	//delai entre deux mesures
  long nb_mesure;		//nb de mesure a effectuer

  HAL_Init();
  AD5940_MCUResourceInit(0);    /* Initialize resources that AD5940 use, like SPI/GPIO/Interrupt. */
  MCUPlatformInit(0);
  //AD5940PlatformCfg();

  my_printf("Hello AD5940-Build Time:%s %s\r\n", FW_Date, FW_Time);

  serial_reading(&delay_mesure, &nb_pt, &freq_min, &freq_max, &nb_mesure);

  //HAL_UART_Receive(&UartHandle, buffer, 300,1000);
  AD5940_BIA(delay_mesure, nb_pt, freq_min, freq_max, nb_mesure);
}

#define DEBUG_UART                         USART2
#define DEBUG_UART_IRQN                    USART2_IRQn
#define DEBUGUART_CLK_ENABLE()             __HAL_RCC_USART2_CLK_ENABLE()
#define DEBUGUART_GPIO_CLK_ENABLE()        __HAL_RCC_GPIOA_CLK_ENABLE()

/* Definition for AD5940 Pins */
#define DEBUGUART_TX_PIN                   GPIO_PIN_2
#define DEBUGUART_TX_GPIO_PORT             GPIOA
#define DEBUGUART_TX_AF                    GPIO_AF7_USART2

#define DEBUGUART_RX_PIN                   GPIO_PIN_3
#define DEBUGUART_RX_GPIO_PORT             GPIOA
#define DEBUGUART_RX_AF                    GPIO_AF7_USART2




void Error_Handler(void){
  while(1);
}
/**
  * @brief SPI MSP Initialization
  *        This function configures the hardware resources used in this example:
  *           - Peripheral's clock enable
  *           - Peripheral's GPIO Configuration
  * @param husart: SPI handle pointer
  * @retval None
  */
//void HAL_UART_MspInit(UART_HandleTypeDef *husart)
//{
//  GPIO_InitTypeDef  GPIO_InitStruct;
//
//  if(husart->Instance == DEBUG_UART)
//  {
//    /*##-1- Enable peripherals and GPIO Clocks #################################*/
//    /* Enable GPIO TX/RX clock */
//    DEBUGUART_GPIO_CLK_ENABLE();
//    /* Enable UART clock */
//    DEBUGUART_CLK_ENABLE();
//
//    /*##-2- Configure peripheral GPIO ##########################################*/
//    /* UART TX GPIO pin configuration  */
//    GPIO_InitStruct.Pin       = DEBUGUART_TX_PIN;
//    GPIO_InitStruct.Mode      = GPIO_MODE_AF_PP;
//    GPIO_InitStruct.Pull      = GPIO_PULLUP;
//    GPIO_InitStruct.Speed     = GPIO_SPEED_FREQ_VERY_HIGH;
//    GPIO_InitStruct.Alternate = DEBUGUART_TX_AF;
//    HAL_GPIO_Init(DEBUGUART_TX_GPIO_PORT, &GPIO_InitStruct);
//
//    /* UART RX GPIO pin configuration  */
//    GPIO_InitStruct.Pin = DEBUGUART_RX_PIN;
//    GPIO_InitStruct.Mode      = GPIO_MODE_AF_PP;
//    GPIO_InitStruct.Alternate = DEBUGUART_RX_AF;
//    HAL_GPIO_Init(DEBUGUART_RX_GPIO_PORT, &GPIO_InitStruct);
//  }
//}

/**
  * @brief reading of the rx uart port
  *
  * @param delay_mesure: the delay between two successive measurement
  * @param nb_pt: number of point during a frequency sweep
  * @param freq_min: first frequency of the frequency sweep
  * @param freq_max: last frequency of the frequency sweep
  * @param nb_mesure: le nombre de mesure à effectuer
  */
void serial_reading(long *delay_mesure, uint32_t *nb_pt, float *freq_min, float *freq_max, long *nb_mesure)
{
	//buffer qui accueil la tram
	uint8_t Rx_buffer[29] = {0};

	//on cherche a avoir une trame du type <delay_mesure>;<nb_point>;<freq_min>;<freq_max>;<nb_mesure>.
	//max 9999;999;199999;200000;99999. soit 29 caracteres
	HAL_UART_Receive_DMA(&UartHandle, Rx_buffer, 29);

	//on attend que le buffer soit plein
	while(fini == 0)
	{
		/*
		AD5940_WriteReg(REG_AGPIO_GP0OUT, AGPIO_Pin1);//gpio1 high (led off)
		HAL_Delay(200);
		AD5940_WriteReg(REG_AGPIO_GP0OUT, 0);//gpio1 low (led on)
		HAL_Delay(200);
		*/
		HAL_Delay(1000);
	}
	//AD5940_WriteReg(REG_AGPIO_GP0OUT, AGPIO_Pin1);//gpio1 high (led off)
	//HAL_UART_Transmit(&UartHandle, Rx_buffer, 11, 1000);

	//obtention de delay_mesure
	int i = 0;
	double tampon = 0;
	uint8_t tab[6] = {0};
	while( Rx_buffer[i] != ';')
	{
		tab[i] = Rx_buffer[i] - '0';
		i++;
	}
	for(int j = 0; j<i; j++)
	{
		tampon += tab[i-(j+1)]*pow(10,j);
	}
	*delay_mesure = (long)tampon;

	//obtention de nb_pt
	//int i = 0;
	i++;
	int i_save = i;
	tampon = 0;
	memset(tab, 0, sizeof tab);
	while( Rx_buffer[i] != ';')
	{
		tab[i-i_save] = Rx_buffer[i] - '0';
		i++;
	}
	for(int j = i_save; j<i; j++)
	{
		tampon += tab[i-(j+1)]*pow(10,j-i_save);
	}
	*nb_pt = (uint32_t)tampon;

	//obtention de freq_min
	//int i = 0;
	i++;
	i_save = i;
	tampon = 0;
	memset(tab, 0, sizeof tab);
	while( Rx_buffer[i] != ';')
	{
		tab[i-i_save] = Rx_buffer[i] - '0';
		i++;
	}
	for(int j = i_save; j<i; j++)
	{
		tampon += tab[i-(j+1)]*pow(10,j-i_save);
	}
	*freq_min = (float)tampon;

	//obtention de freq_max
	//int i = 0;
	i++;
	i_save = i;
	tampon = 0;
	memset(tab, 0, sizeof tab);
	while( Rx_buffer[i] != ';')
	{
		tab[i-i_save] = Rx_buffer[i] - '0';
		i++;
	}
	for(int j = i_save; j<i; j++)
	{
		tampon += tab[i-(j+1)]*pow(10,j-i_save);
	}
	*freq_max = (float)tampon;

	//obtention de nb_mesure
	//int i = 0;
	i++;
	i_save = i;
	tampon = 0;
	memset(tab, 0, sizeof tab);
	while( Rx_buffer[i] != '.')
	{
		tab[i-i_save] = Rx_buffer[i] - '0';
		i++;
	}
	for(int j = i_save; j<i; j++)
	{
		tampon += tab[i-(j+1)]*pow(10,j-i_save);
	}
	*nb_mesure = (float)tampon;

}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the CPU, AHB and APB busses clocks
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 100;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB busses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

void MCUPlatformInit(void *pCfg)
{
	HAL_Init();
	SystemClock_Config();
	MX_DMA_Init();
	//HAL_Init();
	/* Init UART */
	UartHandle.Instance        = DEBUG_UART;

	UartHandle.Init.BaudRate   = 230400;
	UartHandle.Init.WordLength = UART_WORDLENGTH_8B;
	UartHandle.Init.StopBits   = UART_STOPBITS_1;
	UartHandle.Init.Parity     = UART_PARITY_NONE;
	UartHandle.Init.HwFlowCtl  = UART_HWCONTROL_NONE;
	UartHandle.Init.Mode       = UART_MODE_TX_RX;
	UartHandle.Init.OverSampling = UART_OVERSAMPLING_16;
	if (HAL_UART_Init(&UartHandle) != HAL_OK)
	{
	  Error_Handler();
	}

//  if(HAL_UART_Init(&UartHandle) != HAL_OK)
//  {
//    /* Initialization Error */
//    return 0;
//  }
//  __HAL_UART_ENABLE_IT(&UartHandle, UART_IT_RXNE);
//  HAL_NVIC_EnableIRQ(DEBUG_UART_IRQN);
//	return 1;

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream5_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream5_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream5_IRQn);

}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  /* Prevent unused argument(s) compilation warning */
  UNUSED(huart);
  /* NOTE: This function should not be modified, when the callback is needed,
           the HAL_UART_RxCpltCallback could be implemented in the user file
   */
  fini = 1;
  //HAL_UART_Transmit(&huart2, Rx_buffer, 10, 1000);
}

//void USART2_IRQHandler(void)
//{
//  //void UARTCmd_Process(char);
//  volatile char c;
//  if (__HAL_UART_GET_FLAG(&UartHandle, UART_FLAG_RXNE))
//  {
//    c = USART2->DR;
//    //UARTCmd_Process(c);
//  }
//}

#include "stdio.h"
#ifdef __ICCARM__
int putchar(int c)
#else
int fputc(int c, FILE *f)
#endif
{
  uint8_t t = c;
  HAL_UART_Transmit(&UartHandle, &t, 1, 1000);
  return c;
}


/**
  * @brief  This function handles SysTick Handler.
  * @param  None
  * @retval None
  */
/* DEJA DANS /Core/Src/stm32f4xx_it.c:185
void SysTick_Handler(void)
{
  HAL_IncTick();
}*/
