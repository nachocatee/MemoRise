import React, { useState } from "react";
import { RouteProp } from "@react-navigation/native";
import { View, Text, Image, Alert, ScrollView, FlatList } from "react-native";
import LinearGradient from "react-native-linear-gradient";
import axios from "axios";
import { useSelector } from "react-redux";

import { RootStackParamList } from "../../../App";
import GoBackHeader from "../../../components/Header/GoBackHeader";
import { styles } from "./GroupStyle";
import SearchInput from "../../../components/SearchInput";
import { BACKEND_URL } from "../../../util/http";
import { RootState } from "../../../store/store";
import { calculateDynamicWidth } from "../../../constants/dynamicSize";
import UserList from "../../../components/UserList";

type InviteUserScreenProps = {
  route: RouteProp<RootStackParamList, "InviteUser">;
};

type UserData = [
  {
    userSeq: number;
    nickname: string;
    email: string;
    profile: string;
    invited: boolean;
  }
];

const InviteUserScreen: React.FC<InviteUserScreenProps> = ({ route }) => {
  const { teamSeq, teamName } = route.params;
  const userId = useSelector((state: RootState) => state.userInfo.id);

  // 유저 검색
  const [userList, setUserList] = useState<UserData | null>(null);
  const [searchKeyword, setSearchKeyword] = useState("");

  const getUserList = async () => {
    if (searchKeyword) {
      try {
        const res = await axios({
          method: "GET",
          // url: BACKEND_URL + `/teams/${teamSeq}/invite/${userId}`,
          url: BACKEND_URL + `/teams/${teamSeq}/invite/26`, // 더미 데이터
          params: {
            keyword: searchKeyword,
          },
        });
        console.log(res.data);
        setUserList(res.data);
        setSearchKeyword("");
      } catch (err) {
        console.log(err);
      }
    } else {
      Alert.alert("검색어를 입력해 주세요!");
    }
  };

  const searchInputHandler = (enteredText: string) => {
    setSearchKeyword(enteredText);
  };

  return (
    <LinearGradient
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <GoBackHeader />
      <View style={styles.container}>
        <View style={styles.titleContainer}>
          <Text style={styles.title}>{teamName}</Text>
        </View>
        <View style={{ marginTop: calculateDynamicWidth(25) }}>
          <SearchInput
            onChangeText={searchInputHandler}
            value={searchKeyword}
            onSubmitEditing={getUserList}
          />
        </View>
        {userList ? (
          <FlatList
            data={userList}
            keyExtractor={(item) => item.userSeq.toString()}
            renderItem={({ item }) => (
              <UserList profileUri={item.profile} nickname={item.nickname} />
            )}
          />
        ) : (
          <View style={styles.userEmptyContainer}>
            <Image
              style={styles.userEmptyIcon}
              source={require("../../../assets/icons/user_empty.png")}
            />
            <Text style={styles.empty}>검색 결과가 없습니다</Text>
          </View>
        )}
      </View>
    </LinearGradient>
  );
};

export default InviteUserScreen;
