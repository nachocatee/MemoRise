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
import { LogBox } from "react-native";

import LandingScreen from "./screens/Landing/LandingScreen";
import LoginScreen from "./screens/User/LoginScreen";
import SignUpScreen from "./screens/User/SignUpScreen";
import MainScreen from "./screens/Main/MainScreen";
import MenuScreen from "./screens/Menu/MenuScreen";
import MyGroupScreen from "./screens/MenuContents/Group/MyGroupScreen";
import FindGroupScreen, {
  GroupData,
} from "./screens/MenuContents/Group/FindGroupScreen";
import ModifyInfoScreen from "./screens/User/ModifyInfoScreen";
import { store } from "./store/store";
import MakeGroupScreen from "./screens/MenuContents/Group/MakeGroupScreen";
import NotFoundScreen from "./screens/NotFound/NotFoundScreen";
import GroupDetailScreen from "./screens/MenuContents/Group/GroupDetailScreen";
import GroupSettingScreen from "./screens/MenuContents/Group/GroupSettingScreen";
import InviteUserScreen from "./screens/MenuContents/Group/InviteUserScreen";
import GroupSearchResultScreen from "./screens/MenuContents/Group/GroupSearchResultScreen";
import MemuMemo from "./screens/MenuContents/Memo/MenuMemo";
import TutorialScreen from "./screens/Tutorial/TutorialScreen";

export type RootStackParamList = {
  Landing: undefined;
  Login: undefined;
  SignUp: undefined;
  Main: undefined;
  Menu: undefined;
  MyGroup: undefined;
  FindGroup: undefined;
  ModifyInfo: undefined;
  MakeGroup: undefined;
  NotFound: undefined;
  MenuMemo: {
    menuStatus: string;
  };
  GroupDetail: {
    teamSeq: number;
    userSeq: number;
  };
  GroupSetting: {
    name: string;
    password: string;
    teamSeq: number;
  };
  InviteUser: {
    teamSeq: number;
    teamName: string;
  };
  GroupSearchResult: { searchResults?: GroupData };
  Tutorial: undefined;
};

LogBox.ignoreLogs([
  "`flexWrap: `wrap`` is not supported with the `VirtualizedList` components.Consider using `numColumns` with `FlatList` instead.",
]);

const Stack = createStackNavigator<RootStackParamList>();

function App(): JSX.Element {
  return (
    <SafeAreaView style={{ flex: 1 }}>
      <StatusBar />
      <Provider store={store}>
        <NavigationContainer>
          <Stack.Navigator screenOptions={{ headerShown: false }}>
            <Stack.Screen name="Landing" component={LandingScreen} />
            <Stack.Screen name="Main" component={MainScreen} />
            <Stack.Screen name="SignUp" component={SignUpScreen} />
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name="Tutorial" component={TutorialScreen} />
            {/* 메뉴 */}
            <Stack.Screen name="Menu" component={MenuScreen} />
            <Stack.Screen name="MyGroup" component={MyGroupScreen} />
            <Stack.Screen name="FindGroup" component={FindGroupScreen} />
            <Stack.Screen name="ModifyInfo" component={ModifyInfoScreen} />
            <Stack.Screen name="MakeGroup" component={MakeGroupScreen} />
            <Stack.Screen name="GroupDetail" component={GroupDetailScreen} />
            <Stack.Screen name="GroupSetting" component={GroupSettingScreen} />
            <Stack.Screen name="InviteUser" component={InviteUserScreen} />
            <Stack.Screen name="MenuMemo" component={MemuMemo} />
            <Stack.Screen
              name="GroupSearchResult"
              component={GroupSearchResultScreen}
            />
            {/* 404에러 */}
            <Stack.Screen name="NotFound" component={NotFoundScreen} />
          </Stack.Navigator>
        </NavigationContainer>
      </Provider>
    </SafeAreaView>
  );
}

export default App;
