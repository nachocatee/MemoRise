/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 */

import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createStackNavigator } from "@react-navigation/stack";
import { SafeAreaView, StatusBar } from "react-native";

import LandingScreen from "./screens/Landing/LandingScreen";
import LoginScreen from "./screens/User/LoginScreen";
import SignUpScreen from "./screens/User/SignUpScreen";
import MainScreen from "./screens/Main/MainScreen";
import MenuScreen from "./screens/Menu/MenuScreen";
import SavedMemoScreen from "./screens/MenuContents/Memo/SavedMemoScreen";
import AllMemoScreen from "./screens/MenuContents/Memo/AllMemoScreen";
import MyMemoScreen from "./screens/MenuContents/Memo/MyMemoScreen";
import MyGroupScreen from "./screens/MenuContents/Group/MyGroupScreen";
import FindGroupScreen from "./screens/MenuContents/Group/FindGroupScreen";
import ModifyInfoScreen from "./screens/User/ModifyInfoScreen";

export type RootStackParamList = {
  Landing: undefined;
  Login: undefined;
  SignUp: undefined;
  Main: undefined;
  Menu: undefined;
  SavedMemo: undefined;
  AllMemo: undefined;
  MyMemo: undefined;
  MyGroup: undefined;
  FindGroup: undefined;
  ModifyInfo: undefined;
};

const Stack = createStackNavigator<RootStackParamList>();

function App(): JSX.Element {
  return (
    <SafeAreaView style={{ flex: 1 }}>
      <StatusBar />
      <NavigationContainer>
        <Stack.Navigator screenOptions={{ headerShown: false }}>
          <Stack.Screen name="SignUp" component={SignUpScreen} />
          <Stack.Screen name="Landing" component={LandingScreen} />
          <Stack.Screen name="Login" component={LoginScreen} />
          <Stack.Screen name="Main" component={MainScreen} />
          {/* 메뉴 */}
          <Stack.Screen name="Menu" component={MenuScreen} />
          <Stack.Screen name="SavedMemo" component={SavedMemoScreen} />
          <Stack.Screen name="AllMemo" component={AllMemoScreen} />
          <Stack.Screen name="MyMemo" component={MyMemoScreen} />
          <Stack.Screen name="MyGroup" component={MyGroupScreen} />
          <Stack.Screen name="FindGroup" component={FindGroupScreen} />
          <Stack.Screen name="ModifyInfo" component={ModifyInfoScreen} />
        </Stack.Navigator>
      </NavigationContainer>
    </SafeAreaView>
  );
}

export default App;
