package com.tjjhtjh.memorise.domain.tag.service.dto.request;

import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.tag.repository.entity.TaggedUser;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.util.List;

@Getter
@NoArgsConstructor
public class TaggedUserRequest {

    private Long taggedUserSeq;
    private User user;
    private Memo memo;
    private List<Long> userSeqList;

    public TaggedUser saveUserToEntity(Memo memo1, User user1){
        return TaggedUser.builder()
                .user(user1)
                .memo(memo1)
                .build();
    }

}
