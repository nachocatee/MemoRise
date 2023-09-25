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
import { Provider } from "react-redux";

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
import { store } from "./store/store";
import MakeGroupScreen from "./screens/MenuContents/Group/MakeGroupScreen";
import ObjectDetection from "./screens/Main/Camera/ObjectDetection";
import ObjectRegistration from "./screens/Main/Camera/ObjectRegistration";
import NotFoundScreen from "./screens/NotFound/NotFoundScreen";
import GroupDetailScreen from "./screens/MenuContents/Group/GroupDetailScreen";
import GroupSettingScreen from "./screens/MenuContents/Group/GroupSettingScreen";
import InviteUserScreen from "./screens/MenuContents/Group/InviteUserScreen";
import CamTestScreen from "./screens/Main/CamTestScreen";

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
  MakeGroup: undefined;
  ObjectDetection: undefined;
  ObjectRegistration: undefined;
  NotFound: undefined;
  GroupDetail: {
    teamSeq: number;
    userSeq: number;
  };
  GroupSetting: undefined;
  InviteUser: undefined;
  CamTestScreen: undefined;
};

const Stack = createStackNavigator<RootStackParamList>();

function App(): JSX.Element {
  return (
    <SafeAreaView style={{ flex: 1 }}>
      <StatusBar />
      <Provider store={store}>
        <NavigationContainer>
          <Stack.Navigator screenOptions={{ headerShown: false }}>
            <Stack.Screen name="CamTestScreen" component={CamTestScreen} />
            <Stack.Screen name="Landing" component={LandingScreen} />
            <Stack.Screen name="Main" component={MainScreen} />
            <Stack.Screen name="ObjectDetection" component={ObjectDetection} />
            <Stack.Screen
              name="ObjectRegistration"
              component={ObjectRegistration}
            />
            <Stack.Screen name="SignUp" component={SignUpScreen} />
            <Stack.Screen name="Login" component={LoginScreen} />
            {/* 메뉴 */}
            <Stack.Screen name="Menu" component={MenuScreen} />
            <Stack.Screen name="SavedMemo" component={SavedMemoScreen} />
            <Stack.Screen name="AllMemo" component={AllMemoScreen} />
            <Stack.Screen name="MyMemo" component={MyMemoScreen} />
            <Stack.Screen name="MyGroup" component={MyGroupScreen} />
            <Stack.Screen name="FindGroup" component={FindGroupScreen} />
            <Stack.Screen name="ModifyInfo" component={ModifyInfoScreen} />
            <Stack.Screen name="MakeGroup" component={MakeGroupScreen} />
            <Stack.Screen name="GroupDetail" component={GroupDetailScreen} />
            <Stack.Screen name="GroupSetting" component={GroupSettingScreen} />
            <Stack.Screen name="InviteUser" component={InviteUserScreen} />
            {/* 404에러 */}
            <Stack.Screen name="NotFound" component={NotFoundScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </Provider>
    </SafeAreaView>
  );
}

export default App;
