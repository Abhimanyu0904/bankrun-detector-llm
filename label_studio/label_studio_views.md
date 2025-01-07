<View>
  <Header value="Tweet Classification Task"></Header>
  <Text name="text" value="$text"></Text>
  <Choices name="entity_classification" toName="text" choice="single" showInLine="true">
   <Choice value="Incorrect Entity" hotkey="a"/>
   <Choice value="Correct Entity" hotkey="d"/>
  </Choices>
</View>

<View>
  <Header value="Tweet Classification Task"></Header>
  <Text name="desc" value="Below is your entity label. If it is incorrect, your risk label will be ignored."></Text>
  <Text name="entity_classification" value="$entity_classification"></Text>
  <Text name="space" value=" "></Text>
  <Text name="text" value="$text"></Text>
  <Choices name="risk_sentiment" toName="text" choice="single" showInLine="true">
   <Choice value="Not Indicative of a Bank Run" hotkey="a"/>
   <Choice value="Indicative of a Bank Run" hotkey="d"/>
  </Choices>
</View>