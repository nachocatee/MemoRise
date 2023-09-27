import React from "react";
import { View, Text } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import { RouteProp, NavigationProp } from "@react-navigation/native";

import GoBackHeader from "../../../components/Header/GoBackHeader";
import { GroupData, GroupDetailParams } from "./FindGroupScreen";

type RootStackParamList = {
  MakeGroup: undefined;
  GroupDetail: GroupDetailParams;
  GroupSearchResult: { searchResults: GroupData };
};

type GroupSearchResultScreenRouteProp = RouteProp<
  RootStackParamList,
  "GroupSearchResult"
>;

type GroupSearchResultScreenNavigationProp = NavigationProp<
  RootStackParamList,
  "GroupSearchResult"
>;

type Props = {
  route?: GroupSearchResultScreenRouteProp;
  navigation?: GroupSearchResultScreenNavigationProp;
};

const GroupSearchResultScreen: React.FC<Props> = ({ route, navigation }) => {
  if (!route || !navigation) return null;

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View>
        <Text>그룹 검색 결과 페이지</Text>
      </View>
    </LinearGradient>
  );
};

export default GroupSearchResultScreen;
