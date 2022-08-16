import {
  ButtonStyle,
  CLISearchListenerRequest,
  CLISearchListenerResponse,
  CLITemplate,
  TemplateFragment
} from '@openfin/workspace'
import { filter, Subscription, take, takeUntil, withLatestFrom } from 'rxjs'
import { format } from 'date-fns'
// TODO - move into common place
import {
  customNumberFormatter,
  significantDigitsNumberFormatter,
  DECIMAL_SEPARATOR
} from '../../../client/src/utils/formatNumber'
import { BASE_URL, VITE_RT_URL } from '../consts'
import { CurrencyPair, Direction, Trade, TradeStatus } from '../generated/TradingGateway'
import { executing$, executionResponse$ } from '../services/executions'
import { getNlpIntent, NlpIntentType } from '../services/nlpService'
import { getPriceForSymbol$, Price, prices$ } from '../services/prices'
import { tradesStream$ } from '../services/trades'
import {
  createButton,
  createContainer,
  createImage,
  createText,
  createTextContainer
} from '../templates'
import { currencyPairs$, getCurencyPair$ } from '../services/currencyPairs'

const MOVEMENT_UP_ICON = `${BASE_URL}/images/icons/up.svg`
const MOVEMENT_DOWN_ICON = `${BASE_URL}/images/icons/down.svg`

const formatSimple = customNumberFormatter()
const formatTo2Digits = significantDigitsNumberFormatter(2)
const formatTo3Digits = significantDigitsNumberFormatter(3)
const formatToMin2IntDigits = customNumberFormatter({
  minimumIntegerDigits: 2
})

const getPriceBits = (price: number, currencyPair: CurrencyPair) => {
  const { ratePrecision, pipsPosition } = currencyPair
  const rateString = price.toFixed(ratePrecision)
  const [wholeNumber, fractions_] = rateString.split('.')
  const fractions = fractions_ || '00000'

  const pip = formatToMin2IntDigits(Number(fractions.substring(pipsPosition - 2, pipsPosition)))
  const tenth = formatSimple(Number(fractions.substring(pipsPosition, pipsPosition + 1)))

  const bigFigureNumber = Number(wholeNumber + '.' + fractions.substring(0, pipsPosition - 2))
  let bigFigure =
    bigFigureNumber < 1 && pipsPosition === 4
      ? formatTo2Digits(bigFigureNumber)
      : formatTo3Digits(bigFigureNumber)
  if (bigFigureNumber === Math.floor(bigFigureNumber)) bigFigure += DECIMAL_SEPARATOR

  return {
    bigFigure,
    pip,
    tenth
  }
}

const getBaseSpotTemplate = (price: Price): TemplateFragment => {
  const getPriceItem = (direction: 'bid' | 'ask') =>
    createContainer(
      'column',
      [
        createText(`${direction}Label`, 10, { opacity: 0.59, textTransform: 'uppercase' }),
        createContainer(
          'row',
          [
            createText(`${direction}BigFigure`, 14, {}),
            createText(`${direction}Pip`, 36, { lineHeight: 1 }),
            createText(`${direction}Tenth`, 14, {})
          ],
          { alignItems: 'baseline', marginTop: '-14px' }
        )
      ],
      { width: '35%' }
    )

  return createContainer('column', [
    createContainer(
      'row',
      [createText('symbol', 13, { fontWeight: 'bold' }), createText('date', 11, { opacity: 0.59 })],
      {
        justifyContent: 'space-between'
      }
    ),
    createContainer(
      'row',
      [
        getPriceItem('bid'),
        createContainer(
          'column',
          [
            createImage('movementUp', 'Price Movement Up', {
              width: '10px',
              height: '10px',
              visibility: price.lastMove === 'up' ? 'visible' : 'hidden'
            }),
            createText('spread', 12, {}),
            createImage('movementDown', 'Price Movement Down', {
              width: '10px',
              height: '10px',
              visibility: price.lastMove === 'down' ? 'visible' : 'hidden'
            })
          ],
          {
            alignItems: 'center'
          }
        ),
        getPriceItem('ask')
      ],
      {
        justifyContent: 'space-around',
        margin: '20px 0'
      }
    )
  ])
}

const getSpotTemplate = (actions: { launch: string }, price: Price): TemplateFragment => {
  return createContainer(
    'column',
    [
      getBaseSpotTemplate(price),
      createContainer(
        'row',
        [createButton(ButtonStyle.Secondary, 'launchButton', actions.launch, { fontSize: '12px' })],
        { justifyContent: 'flex-end', paddingTop: '10px' }
      )
    ],
    {
      padding: '10px'
    }
  )
}

const getBaseSpotData = (price: Price, currencyPair: CurrencyPair) => {
  const { symbol, bid, ask, valueDate, spread } = price
  const bidPriceBits = getPriceBits(bid, currencyPair)
  const askPriceBits = getPriceBits(ask, currencyPair)

  return {
    symbol,
    bidBigFigure: bidPriceBits.bigFigure,
    bidPip: bidPriceBits.pip,
    bidTenth: bidPriceBits.tenth,
    askBigFigure: askPriceBits.bigFigure,
    askPip: askPriceBits.pip,
    askTenth: askPriceBits.tenth,
    date: `SPT (${format(new Date(valueDate), 'dd MMM').toUpperCase()})`,
    askLabel: 'Buy',
    bidLabel: 'Sell',
    spread,
    movementUp: MOVEMENT_UP_ICON,
    movementDown: MOVEMENT_DOWN_ICON
  }
}

const constructSpotResult = (price: Price, currencyPair: CurrencyPair) => {
  const { symbol } = price
  const LAUNCH_ACTION = `Launch ${symbol} tile`
  const TRADE_ACTION = `Trade ${symbol}`

  return {
    key: `spot-${symbol}`,
    title: symbol,
    label: 'Currency Pair',
    data: {
      symbol,
      manifestType: 'url',
      manifest: `${VITE_RT_URL}/spot/${symbol}`
    },
    actions: [
      { name: LAUNCH_ACTION, hotkey: 'enter' },
      { name: TRADE_ACTION, hotkey: 'CmdOrCtrl+T' }
    ],
    template: CLITemplate.Custom,
    templateContent: {
      layout: getSpotTemplate({ launch: LAUNCH_ACTION }, price),
      data: {
        ...getBaseSpotData(price, currencyPair),
        launchButton: 'Launch'
      }
    }
  }
}

const constructMarketTemplateContent = (
  prices: Price[],
  currencyPairs: Record<string, CurrencyPair>
) => {
  const data = prices.reduce(
    (acc, cur) => {
      const currencyPair = currencyPairs[cur.symbol]
      const { bigFigure, pip, tenth } = getPriceBits(cur.mid, currencyPair)

      return {
        ...acc,
        [`${cur.symbol}Label`]: cur.symbol,
        [`${cur.symbol}BigFigure`]: bigFigure,
        [`${cur.symbol}Pip`]: pip,
        [`${cur.symbol}Tenth`]: tenth,
        [`${cur.symbol}Movement`]: cur.lastMove
      }
    },
    {
      movementUp: MOVEMENT_UP_ICON,
      movementDown: MOVEMENT_DOWN_ICON
    }
  )

  const getTemplate = () => {
    return createContainer(
      'column',
      prices.map(price =>
        createContainer(
          'row',
          [
            createText(`${price.symbol}Label`, 12, { opacity: 0.59, width: '64px' }),
            createContainer(
              'row',
              [
                createText(`${price.symbol}BigFigure`, 10, {}),
                createText(`${price.symbol}Pip`, 22, { lineHeight: 1 }),
                createText(`${price.symbol}Tenth`, 10, {})
              ],
              { width: '60px', alignItems: 'baseline' }
            ),
            createImage(
              price.lastMove === 'up' ? 'movementUp' : 'movementDown',
              `Price Movement ${price.lastMove}`,
              {
                width: '10px',
                height: '10px',
                alignSelf: 'center',
                display: price.lastMove === 'none' ? 'none' : 'block'
              }
            )
          ],
          { alignItems: 'flex-end', marginBottom: '6px' }
        )
      ),
      {
        padding: '10px'
      }
    )
  }

  return {
    layout: getTemplate(),
    data
  }
}

const constructTradeExecutionTemplateContent = (
  price: Price,
  currencyPair: CurrencyPair,
  notional: string,
  direction: Direction
) => {
  const layout: TemplateFragment = createContainer(
    'column',
    [
      getBaseSpotTemplate(price),
      createContainer(
        'row',
        [
          createButton(
            ButtonStyle.Secondary,
            'executeButton', //TODO - const
            `Execute`,
            {
              fontSize: '12px'
            }
          )
        ],
        { justifyContent: 'center', paddingTop: '10px' }
      )
    ],
    {
      padding: '10px'
    }
  )

  const data = {
    ...getBaseSpotData(price, currencyPair),
    executeButton: `${direction} - ${notional}`,
    notional,
    executingLoader: 'Executing'
  }

  return {
    layout,
    data
  }
}

const constructTradeExecutedTemplateContent = (trade: Trade) => {
  const fontSize = 12

  if (trade.status === TradeStatus.Done) {
    const inverseTextStyle = { backgroundColor: 'white', color: '#01C38D', fontWeight: 'bold' }
    const fontSize = 12

    const layout: TemplateFragment = createContainer(
      'column',
      [
        createTextContainer([createText('tradeId')], { fontWeight: 'bold', marginBottom: '10px' }),
        createTextContainer([
          createText('direction', fontSize),
          createText('notional', fontSize, inverseTextStyle),
          createText('rateLabel', fontSize),
          createText('rate', fontSize, inverseTextStyle),
          createText('forLabel', fontSize),
          createText('amount', fontSize, { fontWeight: 'bold', fontStyle: 'italic' }),
          createText('settleLabel', fontSize),
          createText('settleDate', fontSize, { fontWeight: 'bold' })
        ])
      ],
      {
        padding: '10px',
        height: '100%',
        justifyContent: 'center',
        textAlign: 'center',
        backgroundColor: '#01C38D',
        color: 'white'
      }
    )

    const base = trade.currencyPair.slice(0, 3)
    const terms = trade.currencyPair.slice(3, 6)
    const data = {
      tradeId: `Trade ID: ${trade.tradeId.toString()}`,
      direction: `You ${trade.direction === Direction.Buy ? 'bought' : 'sold'} `,
      notional: `${base} ${nf.format(trade.notional)}`,
      rateLabel: ' at a rate of ',
      rate: trade.spotRate.toString(),
      forLabel: ' for ',
      amount: `${terms} ${nf.format(trade.notional * trade.spotRate)}`,
      settleLabel: ` settling (Spt) `,
      settleDate: format(new Date(trade.valueDate), 'dd MMM')
    }

    return {
      layout,
      data
    }
  }

  const layout: TemplateFragment = createContainer(
    'column',
    [
      createTextContainer([createText('tradeId')], { fontWeight: 'bold', marginBottom: '10px' }),
      createTextContainer([createText('rejectedLabel', fontSize)])
    ],
    {
      padding: '10px',
      height: '100%',
      justifyContent: 'center',
      textAlign: 'center',
      backgroundColor: '#FF274B',
      color: 'white'
    }
  )

  const data = {
    tradeId: `Trade ID: ${trade.tradeId.toString()}`,
    rejectedLabel: 'Your trade has been rejected'
  }

  return {
    layout,
    data
  }
}

const nf = new Intl.NumberFormat('default')

export const getNlpResults = async (
  query: string,
  request: CLISearchListenerRequest,
  response: CLISearchListenerResponse
) => {
  let loadingRevoked = false
  const intent = await getNlpIntent(query)

  const revokeLoading = () => {
    if (!loadingRevoked) {
      response.revoke('loading')
      loadingRevoked = true
    }
  }

  if (!intent) {
    return revokeLoading()
  }

  console.log('Intent', intent)

  switch (intent.type) {
    case NlpIntentType.SpotQuote: {
      const { symbol } = intent.payload

      if (!symbol) {
        return revokeLoading()
      }

      let sub = getPriceForSymbol$(symbol)
        .pipe(withLatestFrom(getCurencyPair$(symbol)))
        .subscribe(([priceTick, currencyPair]) => {
          const result = constructSpotResult(priceTick, currencyPair)
          revokeLoading()
          response.respond([result])
        })

      request.onClose(() => {
        if (sub) {
          sub.unsubscribe()
        }
      })

      break
    }

    case NlpIntentType.MarketInfo: {
      const result = {
        key: `market`,
        title: 'Market',
        label: 'Live Rates',
        data: {
          manifestType: 'url',
          manifest: `${VITE_RT_URL}/tiles`
        },
        actions: [{ name: `Launch Live Rates`, hotkey: 'enter' }],
        template: CLITemplate.Custom,
        templateContent: {}
      }

      let sub = prices$
        .pipe(withLatestFrom(currencyPairs$))
        .subscribe(([prices, currencyPairs]) => {
          result.templateContent = constructMarketTemplateContent(
            [...prices.values()],
            currencyPairs
          )

          revokeLoading()
          response.respond([result])
        })

      request.onClose(() => {
        if (sub) {
          sub.unsubscribe()
        }
      })

      break
    }

    case NlpIntentType.TradeInfo: {
      let sub = tradesStream$.subscribe(trades => {
        trades.reverse()
        // @ts-ignore
        const trimmedTrades = intent.payload.count ? trades.splice(0, intent.payload.count) : trades
        const results = trimmedTrades.map(trade => ({
          key: `trade-${trade.tradeId}`,
          title: `${trade.tradeId}`,
          label: 'Trade',
          data: {
            manifestType: 'url',
            manifest: `${VITE_RT_URL}/blotter`
          },
          actions: [{ name: `Launch trades`, hotkey: 'enter' }],
          template: CLITemplate.List,
          templateContent: [
            ['Trade ID', trade.tradeId],
            ['Status', trade.status],
            ['Trade Date', trade.tradeDate],
            ['Direction', trade.direction],
            ['CCYCCY', trade.currencyPair],
            ['Deal CCY', trade.dealtCurrency],
            ['Notional', nf.format(trade.notional)],
            ['Rate', trade.spotRate],
            ['Value Date', trade.valueDate],
            ['Trade', trade.tradeName]
          ]
        }))

        revokeLoading()
        response.respond(results)
      })

      request.onClose(() => {
        if (sub) {
          sub.unsubscribe()
        }
      })

      break
    }

    case NlpIntentType.TradeExecution: {
      const { direction, notional, symbol } = intent.payload as any

      if (!symbol) {
        return revokeLoading()
      }

      const key = `trade-execution-${symbol}`
      let subs: Subscription[] = []
      let result: any

      subs.push(
        getPriceForSymbol$(symbol)
          .pipe(
            withLatestFrom(getCurencyPair$(symbol)),
            takeUntil(executing$.pipe(filter(value => !!value)))
          )
          .subscribe(([price, currencyPair]) => {
            const { bid, ask } = price
            const formattedNotional = nf.format(notional)

            const data = {
              manifestType: 'trade-execution',
              currencyPair: symbol,
              spotRate: direction === Direction.Buy ? ask : bid,
              valueDate: new Date().toISOString().substr(0, 10),
              direction,
              notional,
              dealtCurrency: direction === Direction.Buy ? symbol.substr(0, 3) : symbol.substr(3, 3)
            }
            result = {
              key,
              title: `${direction} ${formattedNotional} ${symbol}`,
              label: 'Trade Execution',
              data,
              actions: [{ name: `Execute`, hotkey: 'enter' }],
              // @ts-ignore
              template: CLITemplate.Custom,
              templateContent: constructTradeExecutionTemplateContent(
                price,
                currencyPair,
                formattedNotional,
                direction
              )
            }

            revokeLoading()
            response.respond([result])
          })
      )

      subs.push(
        executing$.pipe(take(1)).subscribe(() => {
          const newResult = { ...result }
          newResult.actions = []
          newResult.templateContent = {
            ...newResult.templateContent,
            layout: createContainer('column', [
              newResult.templateContent.layout,
              createContainer(
                'column',
                [
                  createText('executingLoader', 12, {
                    background: 'rgb(95, 148, 245)',
                    padding: '16px 32px',
                    borderRadius: '32px',
                    margin: 'auto '
                  })
                ],
                {
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  backgroundColor: 'rgba(0,0,0,.3)',
                  zIndex: 2
                }
              )
            ])
          }
          response.respond([newResult])
        })
      )

      subs.push(
        executionResponse$.pipe(take(1)).subscribe(({ trade }) => {
          response.respond([
            {
              key,
              title: `Trade ${trade.status}`,
              label: 'Trade Execution',
              data: {
                manifestType: 'url',
                manifest: `${VITE_RT_URL}/blotter`
              },
              actions: [{ name: `Launch Trades`, hotkey: 'enter' }],
              // @ts-ignore
              template: CLITemplate.Custom,
              templateContent: constructTradeExecutedTemplateContent(trade)
              // templateContent: [
              //   [trade.tradeId, trade.status],
              //   ['Symbol', trade.currencyPair],
              //   ['Notional', nf.format(trade.notional)],
              //   [trade.direction, trade.spotRate],
              //   ['Date', new Date(trade.tradeDate)]
              // ]
            }
          ])
        })
      )

      request.onClose(() => {
        if (subs.length) {
          subs.forEach(sub => sub.unsubscribe())
        }
      })

      break
    }

    default:
      return revokeLoading()
  }
}
