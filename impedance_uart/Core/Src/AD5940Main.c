/*!
 *****************************************************************************
 @file:    AD5940Main.c
 @author:  Neo Xu
 @brief:   Used to control specific application and process data.
 -----------------------------------------------------------------------------

Copyright (c) 2017-2019 Analog Devices, Inc. All Rights Reserved.

This software is proprietary to Analog Devices, Inc. and its licensors.
By using this software you agree to the terms of the associated
Analog Devices Software License Agreement.

*****************************************************************************/
/**
 * @addtogroup AD5940_System_Examples
 * @{
 *  @defgroup BioElec_Example
 *  @{
  */
#include "ad5940.h"
#include "AD5940.h"
#include <stdio.h>
#include "string.h"
#include "math.h"
#include "BodyImpedance.h"
#include "stm32f4xx_hal.h"
#include "stm32f4xx_it.h"
#include "main.h"

//UART_HandleTypeDef UartHandle;

#define APPBUFF_SIZE 512
uint32_t AppBuff[APPBUFF_SIZE];
uint8_t entree = 13;

/* It's your choice here how to do with the data. Here is just an example to print them to UART */
int32_t BIAShowResult(uint32_t *pData, uint32_t DataCount)
{
  float freq;


  fImpPol_Type *pImp = (fImpPol_Type*)pData;
  AppBIACtrl(BIACTRL_GETFREQ, &freq);

  //my_printf("Freq:%f Hz ", freq);
  my_printf("%f;", freq);
  /*Process data*/
  for(int i=0;i<DataCount;i++)
  {
    //my_printf("RzMag: %f Ohm , RzPhase: %f \r\n",pImp[i].Magnitude,pImp[i].Phase*180/MATH_PI);
    my_printf("%f;%f\r\n",pImp[i].Magnitude,pImp[i].Phase*180/MATH_PI);

  }
  return 0;
}

/* Initialize AD5940 basic blocks like clock */
static int32_t AD5940PlatformCfg(void)
{
  CLKCfg_Type clk_cfg;
  FIFOCfg_Type fifo_cfg;
  AGPIOCfg_Type gpio_cfg;

  /* Use hardware reset */
  AD5940_HWReset();
  /* Platform configuration */
  AD5940_Initialize();
  /* Step1. Configure clock */
  clk_cfg.ADCClkDiv = ADCCLKDIV_1;
  clk_cfg.ADCCLkSrc = ADCCLKSRC_HFOSC;
  clk_cfg.SysClkDiv = SYSCLKDIV_1;
  clk_cfg.SysClkSrc = SYSCLKSRC_HFOSC;
  clk_cfg.HfOSC32MHzMode = bFALSE;
  clk_cfg.HFOSCEn = bTRUE;
  clk_cfg.HFXTALEn = bFALSE;
  clk_cfg.LFOSCEn = bTRUE;
  AD5940_CLKCfg(&clk_cfg);
  /* Step2. Configure FIFO and Sequencer*/
  fifo_cfg.FIFOEn = bFALSE;
  fifo_cfg.FIFOMode = FIFOMODE_FIFO;
  fifo_cfg.FIFOSize = FIFOSIZE_4KB;                       /* 4kB for FIFO, The reset 2kB for sequencer */
  fifo_cfg.FIFOSrc = FIFOSRC_DFT;
  fifo_cfg.FIFOThresh = 4;//AppBIACfg.FifoThresh;        /* DFT result. One pair for RCAL, another for Rz. One DFT result have real part and imaginary part */
  AD5940_FIFOCfg(&fifo_cfg);                             /* Disable to reset FIFO. */
  fifo_cfg.FIFOEn = bTRUE;
  AD5940_FIFOCfg(&fifo_cfg);                             /* Enable FIFO here */

  /* Step3. Interrupt controller */

  AD5940_INTCCfg(AFEINTC_1, AFEINTSRC_ALLINT, bTRUE);           /* Enable all interrupt in Interrupt Controller 1, so we can check INTC flags */
  AD5940_INTCCfg(AFEINTC_0, AFEINTSRC_DATAFIFOTHRESH, bTRUE);   /* Interrupt Controller 0 will control GP0 to generate interrupt to MCU */
  AD5940_INTCClrFlag(AFEINTSRC_ALLINT);
  /* Step4: Reconfigure GPIO */
  gpio_cfg.FuncSet = GP6_SYNC|GP5_SYNC|GP4_SYNC|GP2_TRIG|GP1_GPIO|GP0_INT;
  gpio_cfg.InputEnSet = AGPIO_Pin2;
  gpio_cfg.OutputEnSet = AGPIO_Pin0|AGPIO_Pin1|AGPIO_Pin4|AGPIO_Pin5|AGPIO_Pin6;
  gpio_cfg.OutVal = 0;
  gpio_cfg.PullEnSet = 0;

  AD5940_AGPIOCfg(&gpio_cfg);
  AD5940_SleepKeyCtrlS(SLPKEY_UNLOCK);  /* Allow AFE to enter sleep mode. */
  return 0;
}

/* !!Change the application parameters here if you want to change it to none-default value */
void AD5940BIAStructInit(uint32_t nb_pt, float freq_min, float freq_max)
{
  AppBIACfg_Type *pBIACfg;
  SoftSweepCfg_Type sweepCfg;

  AppBIAGetCfg(&pBIACfg);
  sweepCfg = pBIACfg->SweepCfg;

  pBIACfg->SeqStartAddr = 0;
  pBIACfg->MaxSeqLen = 512; /** @todo add checker in function */

  pBIACfg->RcalVal = 200.0; //1500 pour la carte de Valentin
  //pBIACfg->RcalVal = 10000.0;//carte d'eval
  pBIACfg->DftNum = DFTNUM_8192;
  pBIACfg->NumOfData = -1;      /* Never stop until you stop it manually by AppBIACtrl() function */
  pBIACfg->BiaODR = 20;         /* ODR(Sample Rate) 20Hz */
  pBIACfg->FifoThresh = 4;      /* 4 */
  pBIACfg->ADCSinc3Osr = ADCSINC3OSR_2;
  pBIACfg->SinFreq = 20000;

  sweepCfg.SweepEn = bTRUE;
  sweepCfg.SweepPoints = nb_pt;
  sweepCfg.SweepStart = freq_min;
  sweepCfg.SweepStop = freq_max;
  sweepCfg.SweepLog = bTRUE;
  sweepCfg.SweepIndex = 0;

  pBIACfg->SweepCfg = sweepCfg;

}

void AD5940_BIA(long delay_mesure, uint32_t nb_pt, float freq_min, float freq_max, long nb_mesure)
{
  static uint32_t IntCount = 0;
  //static uint32_t count;
  uint32_t temp;

  AD5940PlatformCfg(); /*maintenant dans le main*/
  AD5940_WriteReg(REG_AGPIO_GP0OUT, 0);//gpio1 high (led on)


  AD5940BIAStructInit(nb_pt, freq_min, freq_max); /* Configure your parameters in this function */
  //AD5940BIAFreqCfg(FreqCount);/*configure la frequence de depart*/

  AppBIAInit(AppBuff, APPBUFF_SIZE);    /* Initialize BIA application. Provide a buffer, which is used to store sequencer commands */
  AppBIACtrl(BIACTRL_START, 0);         /* Control BIA measurement to start. Second parameter has no meaning with this command. */
  long origine_temps = HAL_GetTick(); //origine du temps pour effectuer des mesures dans le temps a interval regulier
  long temps = 0;
  long actual_mesure = 0;
  long TimeOut = delay_mesure*1000;

  my_printf("new\r\n");
  AD5940_WriteReg(REG_AGPIO_GP0OUT, AGPIO_Pin1);//gpio1 high (led off)

  while(actual_mesure < nb_mesure)
  {
    /* Check if interrupt flag which will be set when interrupt occurred. */
    if(AD5940_GetMCUIntFlag())
    {
      IntCount++;
      AD5940_ClrMCUIntFlag(); /* Clear this flag */
      temp = APPBUFF_SIZE;
      AppBIAISR(AppBuff, &temp); /* Deal with it and provide a buffer to store data we got */
      BIAShowResult(AppBuff, temp); /* Show the results to UART */

      if(IntCount >= nb_pt)
      {
        IntCount = 0;
        actual_mesure++;
        //AD5940_AGPIOSet(AGPIO_Pin1);
        my_printf("stop\r\n");
        AD5940_WriteReg(REG_AGPIO_GP0OUT, 0);//gpio1 low (led on)

        while( temps - origine_temps < delay_mesure*1000){
        	temps = HAL_GetTick();
        }
        origine_temps += delay_mesure*1000;
        //AD5940_AGPIOClr(AGPIO_Pin1);

        my_printf("new\r\n");
        AD5940_WriteReg(REG_AGPIO_GP0OUT, AGPIO_Pin1);//gpio1 high (led off)
		AppBIAInit(0, 0);    /* Re-initialize BIA application. Because sequences are ready, no need to provide a buffer, which is used to store sequencer commands */
		AppBIACtrl(BIACTRL_START, 0);          /* Control BIA measurement to start. Second parameter has no meaning with this command. */
      }
    }

    /* gestion du timeout, relance l'ad5941 si le delai est trop long  */
    if(temps - origine_temps > 2*TimeOut)
    {
    	AD5940_ClrMCUIntFlag(); /* Clear this flag */
    	my_printf("stop\r\n");
		AD5940_WriteReg(REG_AGPIO_GP0OUT, 0);//gpio1 low (led on)

		HAL_Delay(1000);

		origine_temps = HAL_GetTick();

		my_printf("new\r\n");
		AD5940_WriteReg(REG_AGPIO_GP0OUT, AGPIO_Pin1);//gpio1 high (led off)
		AppBIAInit(0, 0);    /* Re-initialize BIA application. Because sequences are ready, no need to provide a buffer, which is used to store sequencer commands */
		AppBIACtrl(BIACTRL_START, 0);          /* Control BIA measurement to start. Second parameter has no meaning with this command. */
    }

    temps = HAL_GetTick();
  }

  HAL_Delay(1000);
  AppBIACtrl(BIACTRL_SHUTDOWN, 0);

  while(1)
  {
	  AD5940_WriteReg(REG_AGPIO_GP0OUT, AGPIO_Pin1);//gpio1 high (led off)
	  HAL_Delay(500);
	  AD5940_WriteReg(REG_AGPIO_GP0OUT, 0);//gpio1 low (led on)
	  HAL_Delay(500);
  }
}

/**
 * @}
 * @}
 * */

