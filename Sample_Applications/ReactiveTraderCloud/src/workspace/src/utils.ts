import { App } from '@openfin/workspace-platform'
import { firstValueFrom } from 'rxjs'
import { VITE_RT_URL } from './consts'
import { currencyPairSymbols$ } from './services/currencyPairs'

export const getSpotTileApps = async (): Promise<App[]> => {
  const currencyPairs = await firstValueFrom(currencyPairSymbols$)

  return currencyPairs.map(symbol => ({
    appId: `reactive-trader-${symbol}`,
    manifestType: 'url',
    manifest: `${VITE_RT_URL}/spot/${symbol}`,
    title: `${symbol} Spot Tile`,
    icons: [],
    publisher: 'Adaptive Financial Consulting',
    description: `View ${symbol} live rates`
  }))
}
