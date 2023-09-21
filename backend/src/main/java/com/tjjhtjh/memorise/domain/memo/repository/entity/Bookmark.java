package com.tjjhtjh.memorise.domain.memo.repository.entity;

import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import jakarta.persistence.*;
import lombok.*;
import org.hibernate.annotations.DynamicInsert;
import org.hibernate.annotations.DynamicUpdate;
import org.springframework.data.relational.core.mapping.Table;

@DynamicUpdate
@DynamicInsert
@Getter
@NoArgsConstructor(access = AccessLevel.PROTECTED)
@AllArgsConstructor
@Builder
@Entity
@Table(name = "bookmark")
public class Bookmark {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "bookmark_seq")
    private Long bookmarkSeq;
    @ManyToOne(targetEntity = User.class ,fetch = FetchType.LAZY)
    @JoinColumn(name = "user_seq" , referencedColumnName = "user_seq")
    private User user;
    @ManyToOne(targetEntity = Memo.class ,fetch = FetchType.LAZY)
    @JoinColumn(name = "memo_seq" , referencedColumnName = "memo_seq")
    private Memo memo;

}
